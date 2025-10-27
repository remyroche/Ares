"""
Consolidated Regime Feature Generator

This module provides comprehensive feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Consolidates all regime-related features
from statistical, structural trend, volatility, volume, and advanced regime analysis.

Key Features:
- Statistical regime characteristics (distribution shape, persistence, transitions)
- Structural trend regime features (persistence, strength, market structure)
- Volatility regime features (clustering, persistence, transitions)
- Volume regime features (persistence, clustering, price relationships)
- Advanced regime features (entropy, complexity, fractal dimension, Hurst exponent)
- Unified VectorBT optimization for high-performance regime analysis
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

# Third-party imports
import numpy as np
import pandas as pd

# Optional third-party imports
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis, jarque_bera
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    find_peaks = None
    skew = None
    kurtosis = None
    jarque_bera = None

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None

try:
    import vectorbt as vbt
    from vectorbt.generic import (
        # rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,  # VectorBT doesn't have these
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

try:
    import cupy as cp  # type: ignore[import-untyped]
except ImportError:
    cp = None  # type: ignore[assignment]

# Local imports
from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_optimization_system import get_unified_optimization_system, UnifiedOptimizationSystem
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorization_optimizer = None
    get_optimized_feature_pipeline = None
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None
    get_unified_optimization_system = None
    UnifiedOptimizationSystem = None

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

class RegimeStatisticalFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator for statistical regime features optimized for 15m timeframe.

    This generator creates statistical features that help identify market regimes
    through analysis of price distribution characteristics, persistence patterns,
    and statistical properties of returns.

    Key Features:
    - Distribution shape analysis (skewness, kurtosis)
    - Regime persistence through autocorrelation
    - Statistical tests for regime changes
    - VectorBT optimization for high-frequency data

    Parameters:
    - config: FeatureConfig object with generator parameters
    - window: Lookback window for statistical calculations (default: 20)
    - min_periods: Minimum periods required for valid calculations (default: 10)

    Returns:
    - Dict[str, np.ndarray]: Dictionary of statistical regime features

    Example:
        >>> generator = RegimeStatisticalFeatureGenerator()
        >>> features = generator.generate_features(data)
        >>> print(f"Generated {len(features)} statistical regime features")
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizers
        self.vectorbt_rolling_optimizer = None
        self.unified_optimizer = None
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0,
            'feature_generation_time': 0.0,
            'optimization_time': 0.0,
            'memory_usage_mb': 0.0,
            'features_generated': 0,
            'vectorbt_success_rate': 0.0,
            'unified_optimization_success_rate': 0.0,
            'avg_feature_generation_time': 0.0,
            'total_features_generated': 0
        }

        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT Rolling Optimizer
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True,
                    memory_efficient=True
                )

                # Initialize Unified Optimization System
                from src.feature_generation.utils.unified_optimization_system import UnifiedOptimizationConfig
                unified_config = UnifiedOptimizationConfig(
                    enable_normalization=True,
                    enable_scaling=True,
                    enable_vectorization=True,
                    enable_hardware_optimization=getattr(config, 'gpu_accelerated', False),
                    memory_limit_gb=8.0
                )
                self.unified_optimizer = get_unified_optimization_system(unified_config)

                tprint("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for RegimeStatisticalFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_statistical_features",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description="Statistical regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 30, 80],  # 3h, 7.5h, 20h in 15m periods
                "persistence_windows": [8, 20, 64],  # 2h, 5h, 16h
                "distribution_windows": [16, 40, 128],  # 4h, 10h, 32h
                "transition_windows": [4, 12, 32]  # 1h, 3h, 8h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single statistical regime feature as required by the base class."""
        try:
            # Generate all statistical regime features
            features_dict = self.generate_features(data, **kwargs)

            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple statistical feature if no features generated
                if 'close' in data.columns and len(data) > 1:
                    returns = data['close'].pct_change().fillna(0).values
                    return pd.Series(returns, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            tprint(f"_generate_feature: Statistical regime feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate statistical regime features with unified optimization."""
        import time
        start_time = time.time()

        features = {}

        # Validate price data
        if 'close' not in data.columns:
            return features

        close_prices = data['close'].values
        if len(close_prices) < 8:
            return features

        # Apply unified optimization to data if available
        optimization_start = time.time()
        if self.unified_optimizer:
            try:
                optimized_result = self.unified_optimizer.process_features_unified(
                    data,
                    categories=['regime_statistical'],
                    **kwargs
                )
                if optimized_result.success:
                    data = optimized_result.data
                    self.performance_stats['unified_optimizations'] += 1
                    self.performance_stats['optimization_time'] += time.time() - optimization_start
            except Exception as e:
                tprint(f"⚠️ Unified optimization failed: {e}")
                self.performance_stats['optimization_time'] += time.time() - optimization_start

        # 1. Distribution Shape Features
        features.update(self._generate_distribution_features(close_prices, data))

        # 2. Regime Persistence Features
        features.update(self._generate_persistence_features(close_prices, data))

        # 3. Cross-Correlation Features
        features.update(self._generate_correlation_features(close_prices, data))

        # 4. Regime Transition Features
        features.update(self._generate_transition_features(close_prices, data))

        # 5. Statistical Stability Features
        features.update(self._generate_stability_features(close_prices, data))

        # 6. Enhanced VectorBT Features (if available)
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            features.update(self._calculate_enhanced_statistical_features(close_prices, data))
            features.update(self._calculate_vectorbt_optimized_correlations(close_prices, data))
            features.update(self._calculate_vectorbt_advanced_features(close_prices, data))

        # Update performance stats
        feature_generation_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['processing_time'] += feature_generation_time
        self.performance_stats['feature_generation_time'] += feature_generation_time
        self.performance_stats['features_generated'] = len(features)
        self.performance_stats['total_features_generated'] += len(features)

        # Log performance if verbose
        if hasattr(self.config, 'verbose') and self.config.verbose:
            tprint(f"Generated {len(features)} features in {feature_generation_time:.3f}s")

        return features

    def _generate_distribution_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate distribution shape features for regime detection."""
        features = {}
        windows = self.config.parameters["distribution_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Rolling skewness and kurtosis
            skewness = self._calculate_rolling_skewness(prices, window)
            kurtosis = self._calculate_rolling_kurtosis(prices, window)

            # Distribution normality (Jarque-Bera test)
            normality = self._calculate_rolling_normality(prices, window)

            # Pad to match data length
            skewness_padded = np.full(len(data), np.nan)
            kurtosis_padded = np.full(len(data), np.nan)
            normality_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            skewness_padded[window:] = skewness[window:]
            kurtosis_padded[window:] = kurtosis[window:]
            normality_padded[window:] = normality[window:]

            features[f'distribution_skewness_{window}'] = skewness_padded
            features[f'distribution_kurtosis_{window}'] = kurtosis_padded
            features[f'distribution_normality_{window}'] = normality_padded

        return features

    def _generate_persistence_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime persistence features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Price autocorrelation
            autocorr = self._calculate_price_autocorrelation(prices, window)

            # Regime persistence score
            persistence = self._calculate_regime_persistence(prices, window)

            # Pad to match data length
            autocorr_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            autocorr_padded[window:] = autocorr[window:]
            persistence_padded[window:] = persistence[window:]

            features[f'price_autocorr_{window}'] = autocorr_padded
            features[f'regime_persistence_{window}'] = persistence_padded

        return features

    def _generate_correlation_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-correlation features."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Price-volume correlation if volume available
            if 'volume' in data.columns:
                vol_corr = self._calculate_price_volume_correlation(prices, data['volume'].values, window)
                vol_corr_padded = np.full(len(data), np.nan)
                # Ensure vol_corr has the right length
                if len(vol_corr) >= window:
                    vol_corr_padded[window:] = vol_corr[window:]
                else:
                    # If vol_corr is shorter, pad it appropriately
                    start_idx = max(0, len(data) - len(vol_corr))
                    vol_corr_padded[start_idx:] = vol_corr
                features[f'price_volume_corr_{window}'] = vol_corr_padded

            # Price-range correlation if high/low available
            if 'high' in data.columns and 'low' in data.columns:
                range_corr = self._calculate_price_range_correlation(prices, data['high'].values, data['low'].values, window)
                range_corr_padded = np.full(len(data), np.nan)
                # Ensure range_corr has the right length
                if len(range_corr) >= window:
                    range_corr_padded[window:] = range_corr[window:]
                else:
                    # If range_corr is shorter, pad it appropriately
                    start_idx = max(0, len(data) - len(range_corr))
                    range_corr_padded[start_idx:] = range_corr
                features[f'price_range_corr_{window}'] = range_corr_padded

        return features

    def _generate_transition_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(prices) < window * 2:
                continue

            # Regime change detection
            regime_change = self._detect_regime_changes(prices, window)

            # Transition probability
            transition_prob = self._calculate_transition_probability(prices, window)

            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window*2 onwards
            change_padded[window*2:] = regime_change[window*2:]
            prob_padded[window*2:] = transition_prob[window*2:]

            features[f'regime_change_{window}'] = change_padded
            features[f'transition_prob_{window}'] = prob_padded

        return features

    def _generate_stability_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Statistical stability
            stability = self._calculate_statistical_stability(prices, window)

            # Regime consistency
            consistency = self._calculate_regime_consistency(prices, window)

            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            stability_padded[window:] = stability[window:]
            consistency_padded[window:] = consistency[window:]

            features[f'statistical_stability_{window}'] = stability_padded
            features[f'regime_consistency_{window}'] = consistency_padded

        return features

    def _calculate_rolling_skewness(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        if self.vectorbt_optimizer:
            try:
                skewness = self.vectorbt_rolling_optimizer.rolling_skew(prices_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling skew failed: {e}, using pandas fallback")
                skewness = prices_series.rolling(window).skew()
        else:
            skewness = prices_series.rolling(window).skew()

        return skewness.fillna(0).values

    def _calculate_rolling_kurtosis(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        if self.vectorbt_optimizer:
            try:
                kurtosis = self.vectorbt_rolling_optimizer.rolling_kurt(prices_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling kurt failed: {e}, using pandas fallback")
                kurtosis = prices_series.rolling(window).kurt()
        else:
            kurtosis = prices_series.rolling(window).kurt()

        return kurtosis.fillna(0).values

    def _calculate_rolling_normality(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling normality test - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        if self.vectorbt_optimizer:
            try:
                normality = self.vectorbt_rolling_optimizer.rolling_apply(
                    prices_series,
                    lambda x: jarque_bera(x)[1] if len(x) >= 4 else 0,  # p-value
                    window
                )
            except Exception as e:
                tprint(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                normality = prices_series.rolling(window).apply(
                    lambda x: jarque_bera(x)[1] if len(x) >= 4 else 0,
                    raw=False
                )
        else:
            normality = prices_series.rolling(window).apply(
                lambda x: jarque_bera(x)[1] if len(x) >= 4 else 0,
                raw=False
            )

        return normality.fillna(0).values

    def _calculate_price_autocorrelation(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate price autocorrelation - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series)

        if self.vectorbt_optimizer:
            try:
                autocorr = self.vectorbt_rolling_optimizer.rolling_apply(
                    price_changes,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    window
                )
            except Exception as e:
                tprint(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                autocorr = price_changes.rolling(window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                )
        else:
            autocorr = price_changes.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            )

        return autocorr.fillna(0).values

    def _calculate_regime_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime persistence - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling mean and std
        if self.vectorbt_optimizer:
            try:
                rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window)
                rolling_std = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling operations failed: {e}, using pandas fallback")
                rolling_mean = prices_series.rolling(window).mean()
                rolling_std = prices_series.rolling(window).std()
        else:
            rolling_mean = prices_series.rolling(window).mean()
            rolling_std = prices_series.rolling(window).std()

        # Persistence based on coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        persistence = (1 - cv).clip(0, 1)

        return persistence.fillna(0).values

    def _calculate_price_volume_correlation(self, prices: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate price-volume correlation - OPTIMIZED VECTORBT."""
        if len(prices) < window or len(volume) < window:
            return np.zeros(len(prices))

        prices_series = pd.Series(prices)
        volume_series = pd.Series(volume)

        if self.vectorbt_optimizer:
            try:
                correlation = self.vectorbt_rolling_optimizer.rolling_corr(prices_series, volume_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling corr failed: {e}, using pandas fallback")
                correlation = prices_series.rolling(window).corr(volume_series)
        else:
            correlation = prices_series.rolling(window).corr(volume_series)

        # Ensure the correlation array has the same length as the input data
        correlation_values = correlation.fillna(0).values
        if len(correlation_values) != len(prices):
            # Pad or truncate to match input length
            if len(correlation_values) < len(prices):
                padded = np.zeros(len(prices))
                padded[:len(correlation_values)] = correlation_values
                return padded
            else:
                return correlation_values[:len(prices)]
        
        return correlation_values

    def _calculate_price_range_correlation(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
        """Calculate price-range correlation - OPTIMIZED VECTORBT."""
        if len(prices) < window or len(high) < window or len(low) < window:
            return np.zeros(len(prices))

        prices_series = pd.Series(prices)
        range_series = pd.Series(high - low)

        if self.vectorbt_optimizer:
            try:
                correlation = self.vectorbt_rolling_optimizer.rolling_corr(prices_series, range_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling corr failed: {e}, using pandas fallback")
                correlation = prices_series.rolling(window).corr(range_series)
        else:
            correlation = prices_series.rolling(window).corr(range_series)

        # Ensure the correlation array has the same length as the input data
        correlation_values = correlation.fillna(0).values
        if len(correlation_values) != len(prices):
            # Pad or truncate to match input length
            if len(correlation_values) < len(prices):
                padded = np.zeros(len(prices))
                padded[:len(correlation_values)] = correlation_values
                return padded
            else:
                return correlation_values[:len(prices)]
        
        return correlation_values

    def _detect_regime_changes(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Detect regime changes - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling statistics for both windows
        if self.vectorbt_optimizer:
            try:
                rolling_mean1 = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window)
                rolling_mean2 = rolling_mean1.shift(-window)
            except Exception as e:
                tprint(f"VectorBT rolling operations failed: {e}, using pandas fallback")
                rolling_mean1 = prices_series.rolling(window).mean()
                rolling_mean2 = rolling_mean1.shift(-window)
        else:
            rolling_mean1 = prices_series.rolling(window).mean()
            rolling_mean2 = rolling_mean1.shift(-window)

        # Calculate change ratios
        change_ratios = ((rolling_mean2 - rolling_mean1).abs() / (rolling_mean1 + 1e-8)).fillna(0)
        changes = (change_ratios > 0.3).astype(int)  # 30% change threshold

        return changes.values

    def _calculate_transition_probability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate transition probability - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series)

        if self.vectorbt_optimizer:
            try:
                rolling_std = self.vectorbt_rolling_optimizer.rolling_std(price_changes, window)
                rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(price_changes.abs(), window)
            except Exception as e:
                tprint(f"VectorBT rolling operations failed: {e}, using pandas fallback")
                rolling_std = price_changes.rolling(window).std()
                rolling_mean = price_changes.abs().rolling(window).mean()
        else:
            rolling_std = price_changes.rolling(window).std()
            rolling_mean = price_changes.abs().rolling(window).mean()

        # Transition probability based on volatility changes
        transition_prob = rolling_std / (rolling_mean + 1e-8)
        transition_prob = transition_prob.clip(0, 1)

        return transition_prob.fillna(0).values

    def _calculate_statistical_stability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical stability - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        if self.vectorbt_optimizer:
            try:
                rolling_std = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window)
                rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window)
            except Exception as e:
                tprint(f"VectorBT rolling operations failed: {e}, using pandas fallback")
                rolling_std = prices_series.rolling(window).std()
                rolling_mean = prices_series.rolling(window).mean()
        else:
            rolling_std = prices_series.rolling(window).std()
            rolling_mean = prices_series.rolling(window).mean()

        # Stability based on coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)

        return stability.fillna(0).values

    def _calculate_regime_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime consistency - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling quantiles for consistency
        if self.vectorbt_optimizer:
            try:
                q25 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, window, q=0.25)
                q75 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, window, q=0.75)
            except Exception as e:
                tprint(f"VectorBT rolling quantile failed: {e}, using pandas fallback")
                q25 = prices_series.rolling(window).quantile(0.25)
                q75 = prices_series.rolling(window).quantile(0.75)
        else:
            q25 = prices_series.rolling(window).quantile(0.25)
            q75 = prices_series.rolling(window).quantile(0.75)

        # Consistency based on interquartile range
        iqr = q75 - q25
        rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window) if self.vectorbt_optimizer else prices_series.rolling(window).mean()
        consistency = 1 - (iqr / (rolling_mean + 1e-8))
        consistency = consistency.clip(0, 1)

        return consistency.fillna(0).values

    def _calculate_enhanced_statistical_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate enhanced statistical features using VectorBT native functions."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)

            # Use VectorBT native functions for enhanced calculations
            if self.vectorbt_optimizer:
                # Rolling quantiles for distribution analysis
                q10 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, 20, q=0.1)
                q90 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, 20, q=0.9)

                # Rolling skewness and kurtosis for distribution shape
                rolling_skew = self.vectorbt_rolling_optimizer.rolling_skew(prices_series, 20)
                rolling_kurt = self.vectorbt_rolling_optimizer.rolling_kurt(prices_series, 20)

                # Enhanced statistical features
                features['price_quantile_range'] = (q90 - q10).fillna(0).values
                features['price_skewness_enhanced'] = rolling_skew.fillna(0).values
                features['price_kurtosis_enhanced'] = rolling_kurt.fillna(0).values

                # Rolling rank for position analysis
                rolling_rank = self.vectorbt_rolling_optimizer.rolling_rank(prices_series, 20)
                features['price_rank'] = rolling_rank.fillna(0).values

        except Exception as e:
            tprint(f"Enhanced statistical features calculation failed: {e}")

        return features

    def _calculate_vectorbt_optimized_correlations(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate rolling correlations using VectorBT optimization."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)
            from ...utils.error_handling import safe_diff
            price_changes = safe_diff(prices_series)

            if self.vectorbt_optimizer:
                # Rolling autocorrelation using VectorBT
                rolling_autocorr = self.vectorbt_rolling_optimizer.rolling_apply(
                    price_changes,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
20
                )
                features['price_autocorrelation_enhanced'] = rolling_autocorr.fillna(0).values

                # Rolling correlation with lagged values
                if len(price_changes) > 40:
                    lagged_changes = price_changes.shift(1)
                    rolling_corr = self.vectorbt_rolling_optimizer.rolling_corr(
                        price_changes, lagged_changes, 20
                    )
                    features['price_lag_correlation_enhanced'] = rolling_corr.fillna(0).values

        except Exception as e:
            tprint(f"VectorBT correlation calculation failed: {e}")

        return features

    def _calculate_vectorbt_advanced_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate advanced features using VectorBT native functions."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)

            if self.vectorbt_optimizer:
                # Use VectorBT's winsorize function for outlier handling
                winsorized_prices = winsorize(prices_series, limits=(0.05, 0.05))

                # Calculate rolling statistics on winsorized data
                rolling_mean_win = self.vectorbt_rolling_optimizer.rolling_mean(winsorized_prices, 20)
                rolling_std_win = self.vectorbt_rolling_optimizer.rolling_std(winsorized_prices, 20)

                features['winsorized_mean'] = rolling_mean_win.fillna(0).values
                features['winsorized_std'] = rolling_std_win.fillna(0).values

                # Use VectorBT's clip function for bounded calculations
                clipped_prices = clip(prices_series,
                                    prices_series.quantile(0.01),
                                    prices_series.quantile(0.99))

                # Calculate rolling quantiles using VectorBT
                rolling_q25 = self.vectorbt_rolling_optimizer.rolling_quantile(clipped_prices, 20, q=0.25)
                rolling_q75 = self.vectorbt_rolling_optimizer.rolling_quantile(clipped_prices, 20, q=0.75)

                features['clipped_iqr'] = (rolling_q75 - rolling_q25).fillna(0).values

                # Use VectorBT's zscore function for normalization
                zscored_prices = zscore(prices_series, axis=0)
                rolling_zscore_mean = self.vectorbt_rolling_optimizer.rolling_mean(zscored_prices, 20)

                features['rolling_zscore_mean'] = rolling_zscore_mean.fillna(0).values

        except Exception as e:
            tprint(f"VectorBT advanced features calculation failed: {e}")

        return features

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (self.vectorbt_optimizer is not None and
                len(data) >= 100 and
                VECTORBT_AVAILABLE)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        # Add VectorBT optimizer stats if available
        if self.vectorbt_optimizer:
            vectorbt_stats = self.vectorbt_rolling_optimizer.get_performance_stats()
            stats['vectorbt_stats'] = vectorbt_stats

        # Add unified optimizer stats if available
        if self.unified_optimizer:
            unified_stats = self.unified_optimizer.get_performance_report()
            stats['unified_stats'] = unified_stats

        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0
            stats['unified_optimization_rate'] = stats['unified_optimizations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0
        }

        if self.vectorbt_optimizer:
            self.vectorbt_rolling_optimizer.reset_stats()

class RegimeStructuralTrendFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for structural trend regime features optimized for 15m timeframe."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizers
        self.vectorbt_rolling_optimizer = None
        self.unified_optimizer = None
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0,
            'feature_generation_time': 0.0,
            'optimization_time': 0.0,
            'memory_usage_mb': 0.0,
            'features_generated': 0,
            'vectorbt_success_rate': 0.0,
            'unified_optimization_success_rate': 0.0,
            'avg_feature_generation_time': 0.0,
            'total_features_generated': 0
        }

        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT Rolling Optimizer
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True,
                    memory_efficient=True
                )

                # Initialize Unified Optimization System
                from src.feature_generation.utils.unified_optimization_system import UnifiedOptimizationConfig
                unified_config = UnifiedOptimizationConfig(
                    enable_normalization=True,
                    enable_scaling=True,
                    enable_vectorization=True,
                    enable_hardware_optimization=getattr(config, 'gpu_accelerated', False),
                    memory_limit_gb=8.0
                )
                self.unified_optimizer = get_unified_optimization_system(unified_config)

                tprint("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for RegimeStructuralTrendFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_structural_trend_features",
            category=FeatureCategory.TREND,
            description="Structural trend regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=40,  # 10 hours in 15m periods
            min_lookback=8,       # 2 hours minimum
            max_lookback=160,     # 40 hours maximum
            parameters={
                "structural_windows": [20, 60, 160],  # 5h, 15h, 40h in 15m periods
                "persistence_windows": [16, 40, 128],  # 4h, 10h, 32h
                "transition_windows": [8, 20, 64],  # 2h, 5h, 16h
                "structure_windows": [24, 60, 192]  # 6h, 15h, 48h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single structural trend feature as required by the base class."""
        try:
            # Generate all structural trend features
            features_dict = self.generate_features(data, **kwargs)

            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple trend feature if no features generated
                if 'close' in data.columns and len(data) > 1:
                    trend_feature = data['close'].pct_change().fillna(0).values
                    return pd.Series(trend_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            tprint(f"_generate_feature: Structural trend feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate structural trend regime features with unified optimization."""
        import time
        start_time = time.time()

        features = {}

        # Validate price data
        if 'close' not in data.columns:
            return features

        close_prices = data['close'].values
        if len(close_prices) < 8:
            return features

        # Apply unified optimization to data if available
        optimization_start = time.time()
        if self.unified_optimizer:
            try:
                optimized_result = self.unified_optimizer.process_features_unified(
                    data,
                    categories=['regime_structural_trend'],
                    **kwargs
                )
                if optimized_result.success:
                    data = optimized_result.data
                    self.performance_stats['unified_optimizations'] += 1
                    self.performance_stats['optimization_time'] += time.time() - optimization_start
            except Exception as e:
                tprint(f"⚠️ Unified optimization failed: {e}")
                self.performance_stats['optimization_time'] += time.time() - optimization_start

        # 1. Structural Trend Persistence
        features.update(self._generate_structural_persistence_features(close_prices, data))

        # 2. Trend Regime Strength
        features.update(self._generate_trend_strength_features(close_prices, data))

        # 3. Market Structure Features
        features.update(self._generate_market_structure_features(close_prices, data))

        # 4. Trend Regime Transitions
        features.update(self._generate_trend_transition_features(close_prices, data))

        # 5. Trend Regime Stability
        features.update(self._generate_trend_stability_features(close_prices, data))

        # 6. Enhanced VectorBT Features (if available)
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            features.update(self._calculate_enhanced_trend_features(close_prices, data))
            features.update(self._calculate_vectorbt_optimized_correlations(close_prices, data))
            features.update(self._calculate_vectorbt_advanced_features(close_prices, data))

        # Update performance stats
        feature_generation_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['processing_time'] += feature_generation_time
        self.performance_stats['feature_generation_time'] += feature_generation_time
        self.performance_stats['features_generated'] = len(features)
        self.performance_stats['total_features_generated'] += len(features)

        # Log performance if verbose
        if hasattr(self.config, 'verbose') and self.config.verbose:
            tprint(f"Generated {len(features)} features in {feature_generation_time:.3f}s")

        return features

    def _generate_structural_persistence_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate structural trend persistence features."""
        features = {}
        windows = self.config.parameters["structural_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Structural trend persistence
            trend_persistence = self._calculate_structural_trend_persistence(prices, window)

            # Trend direction consistency
            direction_consistency = self._calculate_trend_direction_consistency(prices, window)

            # Trend regime persistence
            regime_persistence = self._calculate_trend_regime_persistence(prices, window)

            # Pad to match data length
            persistence_padded = np.full(len(data), np.nan)
            direction_padded = np.full(len(data), np.nan)
            regime_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            persistence_padded[window:] = trend_persistence[window:]
            direction_padded[window:] = direction_consistency[window:]
            regime_padded[window:] = regime_persistence[window:]

            features[f'structural_persistence_{window}'] = persistence_padded
            features[f'trend_direction_consistency_{window}'] = direction_padded
            features[f'trend_regime_persistence_{window}'] = regime_padded

        return features

    def _generate_trend_strength_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend strength features."""
        features = {}
        windows = self.config.parameters["structural_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Structural trend strength
            trend_strength = self._calculate_structural_trend_strength(prices, window)

            # Trend acceleration
            trend_acceleration = self._calculate_trend_acceleration(prices, window)

            # Trend regime intensity
            trend_intensity = self._calculate_trend_intensity(prices, window)

            # Pad to match data length
            strength_padded = np.full(len(data), np.nan)
            acceleration_padded = np.full(len(data), np.nan)
            intensity_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            strength_padded[window:] = trend_strength[window:]
            acceleration_padded[window:] = trend_acceleration[window:]
            intensity_padded[window:] = trend_intensity[window:]

            features[f'structural_trend_strength_{window}'] = strength_padded
            features[f'trend_acceleration_{window}'] = acceleration_padded
            features[f'trend_intensity_{window}'] = intensity_padded

        return features

    def _generate_market_structure_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate market structure features."""
        features = {}
        windows = self.config.parameters["structure_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Market structure strength
            structure_strength = self._calculate_market_structure_strength(prices, window)

            # Support/Resistance strength
            support_resistance = self._calculate_support_resistance_strength(prices, window)

            # Market structure consistency
            structure_consistency = self._calculate_market_structure_consistency(prices, window)

            # Pad to match data length
            strength_padded = np.full(len(data), np.nan)
            sr_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            strength_padded[window:] = structure_strength[window:]
            sr_padded[window:] = support_resistance[window:]
            consistency_padded[window:] = structure_consistency[window:]

            features[f'market_structure_strength_{window}'] = strength_padded
            features[f'support_resistance_strength_{window}'] = sr_padded
            features[f'market_structure_consistency_{window}'] = consistency_padded

        return features

    def _generate_trend_transition_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(prices) < window * 2:
                continue

            # Trend regime change detection
            trend_change = self._detect_trend_regime_changes(prices, window)

            # Trend transition probability
            transition_prob = self._calculate_trend_transition_probability(prices, window)

            # Trend regime momentum (structural, not trading)
            trend_momentum = self._calculate_structural_trend_momentum(prices, window)

            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)
            momentum_padded = np.full(len(data), np.nan)

            # trend_change and transition_prob have valid values from index window*2 onwards
            change_padded[window*2:] = trend_change[window*2:]
            prob_padded[window*2:] = transition_prob[window*2:]
            # trend_momentum has valid values from index window onwards
            momentum_padded[window:] = trend_momentum[window:]

            features[f'trend_regime_change_{window}'] = change_padded
            features[f'trend_transition_prob_{window}'] = prob_padded
            features[f'structural_trend_momentum_{window}'] = momentum_padded

        return features

    def _generate_trend_stability_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(prices) < window:
                continue

            # Trend regime stability
            trend_stability = self._calculate_trend_stability(prices, window)

            # Trend persistence score
            persistence_score = self._calculate_trend_persistence_score(prices, window)

            # Trend regime entropy
            trend_entropy = self._calculate_trend_entropy(prices, window)

            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)
            entropy_padded = np.full(len(data), np.nan)

            # Functions return len(prices) with valid values from index window onwards
            stability_padded[window:] = trend_stability[window:]
            persistence_padded[window:] = persistence_score[window:]
            entropy_padded[window:] = trend_entropy[window:]

            features[f'trend_stability_{window}'] = stability_padded
            features[f'trend_persistence_score_{window}'] = persistence_padded
            features[f'trend_entropy_{window}'] = entropy_padded

        return features

    def _calculate_structural_trend_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend persistence - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Pre-calculate rolling statistics using VectorBT
        rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window) if self.vectorbt_optimizer else self._vectorbt_rolling_operation(prices_series, "mean", window)
        rolling_std = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window) if self.vectorbt_optimizer else self._vectorbt_rolling_operation(prices_series, "std", window)

        # OPTIMIZED: Use VectorBT native functions for enhanced calculations
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Calculate rolling linear regression using VectorBT
                rolling_slope = self.vectorbt_rolling_optimizer.rolling_apply(
                    prices_series,
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
                    window
                ).fillna(0)

                # Use VectorBT's scale function for normalization
                normalized_slope = scale(rolling_slope, axis=0)

                # Persistence based on normalized slope consistency
                persistence = (normalized_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)

            except Exception:
                # Fallback to simpler calculation
                rolling_slope = self.vectorbt_rolling_optimizer.rolling_apply(
                    prices_series,
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
                    window
                ).fillna(0)
                persistence = (rolling_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)
        else:
            rolling_slope = self._vectorbt_rolling_operation(prices_series, "apply", window, func=lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0).fillna(0)
            persistence = (rolling_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)

        return persistence.values

    def _calculate_trend_direction_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend direction consistency - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate price changes vectorized
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series)

        # OPTIMIZED: Use VectorBT native functions for enhanced calculations
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT's rank function for direction analysis
                price_ranks = self.vectorbt_rolling_optimizer.rolling_rank(price_changes, window)

                # Calculate direction consistency using ranks
                positive_ranks = (price_ranks > 0.5).astype(int)
                negative_ranks = (price_ranks < 0.5).astype(int)

                positive_changes = self.vectorbt_rolling_optimizer.rolling_sum(positive_ranks, window).fillna(0)
                negative_changes = self.vectorbt_rolling_optimizer.rolling_sum(negative_ranks, window).fillna(0)

            except Exception:
                # Fallback to standard calculation
                positive_changes = self.vectorbt_rolling_optimizer.rolling_sum((price_changes > 0).astype(int), window).fillna(0)
                negative_changes = self.vectorbt_rolling_optimizer.rolling_sum((price_changes < 0).astype(int), window).fillna(0)
        else:
            positive_changes = self._vectorbt_rolling_operation((price_changes > 0).astype(int), "sum", window).fillna(0)
            negative_changes = self._vectorbt_rolling_operation((price_changes < 0).astype(int), "sum", window).fillna(0)

        total_changes = positive_changes + negative_changes

        # Vectorized consistency calculation
        consistency = np.where(
            total_changes > 0,
            np.maximum(positive_changes, negative_changes) / total_changes,
            0
        )

        return consistency

    def _calculate_trend_regime_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend regime persistence - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate price changes for autocorrelation
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series)

        # OPTIMIZED: Use VectorBT rolling apply for autocorrelation
        if self.vectorbt_optimizer:
            persistence = self.vectorbt_rolling_optimizer.rolling_apply(
                price_changes,
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                window
            ).fillna(0)
        else:
            persistence = self._vectorbt_rolling_operation(price_changes, "apply", window, func=lambda x: x.autocorr(lag=1) if len(x) > 1 else 0).fillna(0)

        return persistence.values

    def _calculate_structural_trend_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Pre-calculate rolling statistics using VectorBT
        if self.vectorbt_optimizer:
            rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window)
            rolling_std = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window)
        else:
            rolling_mean = self._vectorbt_rolling_operation(prices_series, "mean", window)
            rolling_std = self._vectorbt_rolling_operation(prices_series, "std", window)

        # Simplified R-squared using variance ratio
        # R² ≈ 1 - (variance_around_trend / total_variance)
        # Approximate trend variance using rolling std
        trend_strength = (1 - (rolling_std / (rolling_mean + 1e-8))).clip(0, 1)

        return trend_strength.fillna(0).values

    def _calculate_trend_acceleration(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend acceleration - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate first and second differences vectorized
        from ...utils.error_handling import safe_diff
        first_diff = safe_diff(prices_series)
        second_diff = safe_diff(first_diff)

        # Rolling acceleration using VectorBT
        if self.vectorbt_optimizer:
            acceleration = self.vectorbt_rolling_optimizer.rolling_mean(second_diff, window)
        else:
            acceleration = self._vectorbt_rolling_operation(second_diff, "mean", window)

        return acceleration.fillna(0).values

    def _calculate_trend_intensity(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend intensity - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling volatility using VectorBT
        if self.vectorbt_optimizer:
            rolling_vol = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window)
        else:
            rolling_vol = self._vectorbt_rolling_operation(prices_series, "std", window)

        price_change = (prices_series - prices_series.shift(window)).abs()

        # Vectorized intensity calculation
        intensity = price_change / (rolling_vol + 1e-8)

        return intensity.fillna(0).values

    def _calculate_market_structure_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling highs and lows using VectorBT
        if self.vectorbt_optimizer:
            rolling_highs = self.vectorbt_rolling_optimizer.rolling_max(prices_series, window)
            rolling_lows = self.vectorbt_rolling_optimizer.rolling_min(prices_series, window)
        else:
            rolling_highs = self._vectorbt_rolling_operation(prices_series, "max", window)
            rolling_lows = self._vectorbt_rolling_operation(prices_series, "min", window)

        # Structure strength based on position within range
        price_range = rolling_highs - rolling_lows
        position = (prices_series - rolling_lows) / (price_range + 1e-8)

        # Strength based on how well-defined the structure is
        strength = (1 - (position - 0.5).abs() * 2).clip(0, 1)

        return strength.fillna(0).values

    def _calculate_support_resistance_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate support/resistance strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling price levels using VectorBT
        if self.vectorbt_optimizer:
            rolling_highs = self.vectorbt_rolling_optimizer.rolling_max(prices_series, window)
            rolling_lows = self.vectorbt_rolling_optimizer.rolling_min(prices_series, window)
        else:
            rolling_highs = self._vectorbt_rolling_operation(prices_series, "max", window)
            rolling_lows = self._vectorbt_rolling_operation(prices_series, "min", window)

        # Strength based on price level consistency (simplified)
        price_range = rolling_highs - rolling_lows
        level_consistency = 1 / (1 + price_range / (prices_series + 1e-8))

        return level_consistency.fillna(0).values

    def _calculate_market_structure_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure consistency - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)

        # Calculate rolling price level consistency using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_rolling_optimizer.rolling_std(prices_series, window)
            rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(prices_series, window)
        else:
            rolling_std = self._vectorbt_rolling_operation(prices_series, "std", window)
            rolling_mean = self._vectorbt_rolling_operation(prices_series, "mean", window)

        # Consistency based on coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)

        return consistency.fillna(0).values

    def _detect_trend_regime_changes(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Detect trend regime changes - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate rolling price changes for trend detection
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(price_series)

        # Calculate rolling means using VectorBT
        if self.vectorbt_optimizer:
            trend1 = self.vectorbt_rolling_optimizer.rolling_mean(price_changes, window)
        else:
            trend1 = self._vectorbt_rolling_operation(price_changes, "mean", window)

        trend2 = trend1.shift(-window)

        # Vectorized change detection using simplified approach
        change_ratios = ((trend2 - trend1).abs() / (trend1.abs() + 1e-8)).fillna(0)
        changes = (change_ratios > 0.5).astype(int)

        return changes.fillna(0).values

    def _calculate_trend_transition_probability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend transition probability - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate rolling price changes for trend analysis
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(price_series)

        # Calculate rolling volatility using VectorBT
        if self.vectorbt_optimizer:
            trend_vol = self.vectorbt_rolling_optimizer.rolling_std(price_changes, window)
            trend_mean = self.vectorbt_rolling_optimizer.rolling_mean(price_changes, window).abs()
        else:
            trend_vol = self._vectorbt_rolling_operation(price_changes, "std", window)
            trend_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()

        # Vectorized transition probability calculation
        transition_prob = np.minimum(1, trend_vol / (trend_mean + 1e-8))

        return transition_prob.fillna(0).values

    def _calculate_structural_trend_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend momentum - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate second differences for momentum
        from ...utils.error_handling import safe_diff
        first_diff = safe_diff(price_series)
        second_diff = safe_diff(first_diff)

        # Rolling momentum using VectorBT
        if self.vectorbt_optimizer:
            momentum = self.vectorbt_rolling_optimizer.rolling_mean(second_diff, window)
        else:
            momentum = self._vectorbt_rolling_operation(second_diff, "mean", window)

        return momentum.fillna(0).values

    def _calculate_trend_stability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend stability - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate rolling coefficient of variation using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_rolling_optimizer.rolling_std(price_series, window)
            rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(price_series, window)
        else:
            rolling_std = self._vectorbt_rolling_operation(price_series, "std", window)
            rolling_mean = self._vectorbt_rolling_operation(price_series, "mean", window)

        # Stability based on low coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)

        return stability.fillna(0).values

    def _calculate_trend_persistence_score(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend persistence score - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate price changes for autocorrelation
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(price_series)

        # OPTIMIZED: Use VectorBT rolling apply for autocorrelation
        if self.vectorbt_optimizer:
            persistence = self.vectorbt_rolling_optimizer.rolling_apply(
                price_changes,
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                window
            ).fillna(0)
        else:
            persistence = self._vectorbt_rolling_operation(price_changes, "apply", window, func=lambda x: x.autocorr(lag=1) if len(x) > 1 else 0).fillna(0)

        return persistence.values

    def _calculate_trend_entropy(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend entropy - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))

        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)

        # Calculate price changes for entropy
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(price_series)

        # Rolling entropy using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_rolling_optimizer.rolling_std(price_changes, window)
            rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(price_changes, window).abs()
        else:
            rolling_std = self._vectorbt_rolling_operation(price_changes, "std", window)
            rolling_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()

        # Entropy proxy using coefficient of variation
        entropy = rolling_std / (rolling_mean + 1e-8)

        return entropy.fillna(0).values

    def _calculate_enhanced_trend_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate enhanced trend features using VectorBT native functions."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)

            # Use VectorBT native functions for enhanced calculations
            if self.vectorbt_optimizer:
                # Rolling quantiles for trend distribution analysis
                q25 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, 20, q=0.25)
                q75 = self.vectorbt_rolling_optimizer.rolling_quantile(prices_series, 20, q=0.75)

                # Rolling skewness and kurtosis for trend shape analysis
                rolling_skew = self.vectorbt_rolling_optimizer.rolling_skew(prices_series, 20)
                rolling_kurt = self.vectorbt_rolling_optimizer.rolling_kurt(prices_series, 20)

                # Enhanced trend features
                features['trend_quartile_range'] = (q75 - q25).fillna(0).values
                features['trend_skewness'] = rolling_skew.fillna(0).values
                features['trend_kurtosis'] = rolling_kurt.fillna(0).values

                # Rolling rank for trend position analysis
                rolling_rank = self.vectorbt_rolling_optimizer.rolling_rank(prices_series, 20)
                features['trend_rank'] = rolling_rank.fillna(0).values

                # Price position within rolling range
                rolling_min = self.vectorbt_rolling_optimizer.rolling_min(prices_series, 20)
                rolling_max = self.vectorbt_rolling_optimizer.rolling_max(prices_series, 20)
                price_position = (prices_series - rolling_min) / (rolling_max - rolling_min + 1e-8)
                features['trend_position'] = price_position.fillna(0.5).values

        except Exception as e:
            tprint(f"Enhanced trend features calculation failed: {e}")

        return features

    def _calculate_vectorbt_optimized_correlations(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate rolling correlations using VectorBT optimization."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)

            # Calculate price changes for correlation analysis
            from ...utils.error_handling import safe_diff
            price_changes = safe_diff(prices_series)

            if self.vectorbt_optimizer:
                # Rolling autocorrelation using VectorBT
                rolling_autocorr = self.vectorbt_rolling_optimizer.rolling_apply(
                    price_changes,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
20
                )
                features['price_autocorrelation'] = rolling_autocorr.fillna(0).values

                # Rolling correlation with lagged values
                if len(price_changes) > 40:
                    lagged_changes = price_changes.shift(1)
                    rolling_corr = self.vectorbt_rolling_optimizer.rolling_corr(
                        price_changes, lagged_changes, 20
                    )
                    features['price_lag_correlation'] = rolling_corr.fillna(0).values

                # Use VectorBT native functions for additional correlations
                if VECTORBT_AVAILABLE:
                    try:
                        # Rolling correlation with volume if available
                        if 'volume' in data.columns and len(data['volume']) > 20:
                            volume_series = data['volume']
                            from ...utils.error_handling import safe_diff
                            volume_changes = safe_diff(volume_series)

                            price_volume_corr = self.vectorbt_rolling_optimizer.rolling_corr(
                                price_changes, volume_changes, 20
                            )
                            features['price_volume_correlation'] = price_volume_corr.fillna(0).values

                        # Rolling correlation with high-low range if available
                        if 'high' in data.columns and 'low' in data.columns:
                            hl_range = data['high'] - data['low']
                            from ...utils.error_handling import safe_diff
                            range_changes = safe_diff(hl_range)

                            price_range_corr = self.vectorbt_rolling_optimizer.rolling_corr(
                                price_changes, range_changes, 20
                            )
                            features['price_range_correlation'] = price_range_corr.fillna(0).values

                    except Exception as e:
                        tprint(f"Additional VectorBT correlation calculations failed: {e}")

        except Exception as e:
            tprint(f"VectorBT correlation calculation failed: {e}")

        return features

    def _calculate_vectorbt_advanced_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate advanced features using VectorBT native functions."""
        features = {}

        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features

        try:
            prices_series = pd.Series(prices)

            if self.vectorbt_optimizer:
                # Use VectorBT's winsorize function for outlier handling
                winsorized_prices = winsorize(prices_series, limits=(0.05, 0.05))

                # Calculate rolling statistics on winsorized data
                rolling_mean_win = self.vectorbt_rolling_optimizer.rolling_mean(winsorized_prices, 20)
                rolling_std_win = self.vectorbt_rolling_optimizer.rolling_std(winsorized_prices, 20)

                features['winsorized_mean'] = rolling_mean_win.fillna(0).values
                features['winsorized_std'] = rolling_std_win.fillna(0).values

                # Use VectorBT's clip function for bounded calculations
                clipped_prices = clip(prices_series,
                                    prices_series.quantile(0.01),
                                    prices_series.quantile(0.99))

                # Calculate rolling quantiles using VectorBT
                rolling_q25 = self.vectorbt_rolling_optimizer.rolling_quantile(clipped_prices, 20, q=0.25)
                rolling_q75 = self.vectorbt_rolling_optimizer.rolling_quantile(clipped_prices, 20, q=0.75)

                features['clipped_iqr'] = (rolling_q75 - rolling_q25).fillna(0).values

                # Use VectorBT's zscore function for normalization
                zscored_prices = zscore(prices_series, axis=0)
                rolling_zscore_mean = self.vectorbt_rolling_optimizer.rolling_mean(zscored_prices, 20)

                features['rolling_zscore_mean'] = rolling_zscore_mean.fillna(0).values

        except Exception as e:
            tprint(f"VectorBT advanced features calculation failed: {e}")

        return features

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (self.vectorbt_optimizer is not None and
                len(data) >= 100 and
                VECTORBT_AVAILABLE)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        # Add VectorBT optimizer stats if available
        if self.vectorbt_optimizer:
            vectorbt_stats = self.vectorbt_rolling_optimizer.get_performance_stats()
            stats['vectorbt_stats'] = vectorbt_stats

        # Add unified optimizer stats if available
        if self.unified_optimizer:
            unified_stats = self.unified_optimizer.get_performance_report()
            stats['unified_stats'] = unified_stats

        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0
            stats['unified_optimization_rate'] = stats['unified_optimizations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0
        }

        if self.vectorbt_optimizer:
            self.vectorbt_rolling_optimizer.reset_stats()

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            # Use VectorBT Rolling Optimizer if available
            if self.vectorbt_optimizer:
                if operation == 'mean':
                    result = self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    result = self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    result = self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    result = self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    result = self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    result = self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    result = self.vectorbt_rolling_optimizer.rolling_apply(data, window, func, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                self.performance_stats['vectorbt_operations'] += 1
                return result
            else:
                # Direct VectorBT usage
                if operation == 'mean':
                    result = rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    result = rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    result = rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    result = rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    result = rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    result = rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    result = rolling_apply(data, func, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                self.performance_stats['vectorbt_operations'] += 1
                return result

        except Exception as e:
            tprint(f"VectorBT operation failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas with VectorBT optimization when available."""
        # Try VectorBT first if available
        if VECTORBT_AVAILABLE and len(data) > 100:  # Use VectorBT for larger datasets
            try:
                if operation == 'mean':
                    return rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return rolling_apply(data, func, window, **kwargs)
            except Exception:
                pass  # Fall back to pandas

        # Pandas fallback
        if operation == 'mean':
            return data.rolling(window).mean()
        elif operation == 'std':
            return data.rolling(window).std()
        elif operation == 'var':
            return data.rolling(window).var()
        elif operation == 'min':
            return data.rolling(window).min()
        elif operation == 'max':
            return data.rolling(window).max()
        elif operation == 'sum':
            return data.rolling(window).sum()
        elif operation == 'apply':
            func = kwargs.pop('func')  # Remove func from kwargs to avoid duplicate argument
            return data.rolling(window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

class RegimeVolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility regime features optimized for 15m timeframe."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizers
        self.vectorbt_rolling_optimizer = None
        self.unified_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True
                )
                self.unified_optimizer = get_unified_optimization_system()
                tprint("✅ VectorBT optimizers initialized for RegimeVolatilityFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Volatility regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 30, 80],  # 3h, 7.5h, 20h in 15m periods
                "persistence_windows": [8, 20, 64],  # 2h, 5h, 16h
                "vol_of_vol_windows": [16, 40, 128],  # 4h, 10h, 32h
                "transition_windows": [4, 12, 32]  # 1h, 3h, 8h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single volatility regime feature as required by the base class."""
        try:
            # Generate all volatility features
            features_dict = self.generate_features(data, **kwargs)

            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple volatility feature if no features generated
                returns = self._get_returns(data)
                if returns is not None and len(returns) > 0:
                    vol_feature = np.abs(returns)  # Simple volatility proxy
                    return pd.Series(vol_feature, index=data.index[1:len(vol_feature)+1])
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            tprint(f"_generate_feature: Volatility feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate volatility regime features."""
        features = {}

        # Get base calculations
        returns = self._get_returns(data)
        if returns is None:
            return features

        # 1. Volatility Regime Persistence
        features.update(self._generate_volatility_persistence_features(returns, data))

        # 2. Volatility Clustering Features
        features.update(self._generate_volatility_clustering_features(returns, data))

        # 3. Volatility-of-Volatility Features
        features.update(self._generate_vol_of_vol_features(returns, data))

        # 4. Volatility Regime Transitions
        features.update(self._generate_volatility_transition_features(returns, data))

        # 5. Volatility Regime Stability
        features.update(self._generate_volatility_stability_features(returns, data))

        return features

    def _get_returns(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate log returns for volatility analysis."""
        if 'close' not in data.columns:
            return None

        close_prices = data['close'].values
        if len(close_prices) < 2:
            return None

        # Use log returns for better volatility regime analysis
        returns = np.diff(np.log(close_prices))
        return returns

    def _generate_volatility_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility persistence features for regime detection."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Rolling volatility
            vol = self._rolling_volatility(returns, window)

            # Volatility persistence (autocorrelation of volatility)
            vol_persistence = self._calculate_volatility_persistence(vol, window // 4)

            # Volatility regime strength
            vol_regime_strength = self._calculate_volatility_regime_strength(vol, window)

            # Pad to match data length
            vol_padded = np.full(len(data), np.nan)
            vol_persistence_padded = np.full(len(data), np.nan)
            vol_strength_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # vol is a rolling result of returns, vol[0] aligns with data[window]
            # vol has length len(returns) - window + 1
            vol_padded[window:window+len(vol)] = vol
            vol_persistence_padded[window:window+len(vol_persistence)] = vol_persistence
            vol_strength_padded[window:window+len(vol_regime_strength)] = vol_regime_strength

            features[f'vol_persistence_{window}'] = vol_persistence_padded
            features[f'vol_regime_strength_{window}'] = vol_strength_padded

        return features

    def _generate_volatility_clustering_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility clustering features."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # GARCH-like volatility clustering
            vol_clustering = self._calculate_volatility_clustering(returns, window)

            # Volatility regime consistency
            vol_consistency = self._calculate_volatility_consistency(returns, window)

            # Pad to match data length
            clustering_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window:] aligns with data[window+1:]
            clustering_padded[window+1:] = vol_clustering[window:]
            consistency_padded[window+1:] = vol_consistency[window:]

            features[f'vol_clustering_{window}'] = clustering_padded
            features[f'vol_consistency_{window}'] = consistency_padded

        return features

    def _generate_vol_of_vol_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility-of-volatility features."""
        features = {}
        windows = self.config.parameters["vol_of_vol_windows"]

        for window in windows:
            if len(returns) < window * 2:
                continue

            # Calculate volatility of volatility
            vol_of_vol = self._calculate_volatility_of_volatility(returns, window)

            # Volatility regime uncertainty
            vol_uncertainty = self._calculate_volatility_uncertainty(returns, window)

            # Pad to match data length
            vol_of_vol_padded = np.full(len(data), np.nan)
            uncertainty_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window:] aligns with data[window+1:]
            vol_of_vol_padded[window*2+1:] = vol_of_vol[window*2:]
            uncertainty_padded[window+1:] = vol_uncertainty[window:]

            features[f'vol_of_vol_{window}'] = vol_of_vol_padded
            features[f'vol_uncertainty_{window}'] = uncertainty_padded

        return features

    def _generate_volatility_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(returns) < window * 2:
                continue

            # Volatility regime change detection
            vol_change = self._detect_volatility_regime_changes(returns, window)

            # Transition probability
            transition_prob = self._calculate_volatility_transition_probability(returns, window)

            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window*2:] aligns with data[window*2+1:]
            change_padded[window*2+1:] = vol_change[window*2:]
            prob_padded[window*2+1:] = transition_prob[window*2:]

            features[f'vol_regime_change_{window}'] = change_padded
            features[f'vol_transition_prob_{window}'] = prob_padded

        return features

    def _generate_volatility_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Volatility regime stability
            vol_stability = self._calculate_volatility_stability(returns, window)

            # Regime persistence score
            persistence_score = self._calculate_regime_persistence_score(returns, window)

            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so vol_stability[window:] aligns with data[window+1:]
            stability_padded[window+1:] = vol_stability[window:]
            persistence_padded[window+1:] = persistence_score[window:]

            features[f'vol_stability_{window}'] = stability_padded
            features[f'regime_persistence_{window}'] = persistence_padded

        return features

    def _rolling_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling volatility - VECTORIZED."""
        if len(returns) < window:
            return np.array([])

        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        vol = self._vectorbt_rolling_operation(returns_series, "std", window).dropna().values

        return vol

    def _calculate_volatility_persistence(self, vol: np.ndarray, lag: int) -> np.ndarray:
        """Calculate volatility persistence using autocorrelation - OPTIMIZED VECTORIZED."""
        if len(vol) < lag + 1:
            return np.zeros(len(vol))

        # OPTIMIZED: Use VectorBT rolling operations for better performance
        vol_series = pd.Series(vol)

        # Calculate volatility changes for autocorrelation
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(vol_series)

        # Use VectorBT rolling apply for autocorrelation calculation
        if self.vectorbt_optimizer:
            try:
                persistence = self.vectorbt_rolling_optimizer.rolling_apply(
                    vol_changes,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
lag+1
                ).fillna(0).values
            except Exception as e:
                tprint(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                persistence = vol_changes.rolling(lag+1).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                ).fillna(0).values
        else:
            # Fallback to pandas
            persistence = vol_changes.rolling(lag+1).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            ).fillna(0).values

        return persistence

    def _calculate_volatility_regime_strength(self, vol: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime strength - VECTORIZED."""
        if len(vol) < window:
            return np.zeros(len(vol))

        # Vectorized regime strength calculation
        vol_series = pd.Series(vol)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)

        # Regime strength based on consistency of volatility level
        vol_consistency = 1.0 - (rolling_std / (rolling_mean + 1e-8))
        strength = vol_consistency.clip(0, 1)

        return strength.fillna(0).values

    def _calculate_volatility_clustering(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate GARCH-like volatility clustering - OPTIMIZED VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))

        # OPTIMIZED: Use VectorBT rolling operations for better performance
        returns_series = pd.Series(returns)
        squared_returns = returns_series ** 2

        # Use VectorBT rolling apply for autocorrelation calculation
        if self.vectorbt_optimizer:
            try:
                clustering = self.vectorbt_rolling_optimizer.rolling_apply(
                    squared_returns,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
window
                ).fillna(0).values
            except Exception as e:
                tprint(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                clustering = squared_returns.rolling(window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                ).fillna(0).values
        else:
            # Fallback to pandas
            clustering = squared_returns.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            ).fillna(0).values

        return clustering

    def _calculate_volatility_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime consistency - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))

        # Vectorized consistency calculation using rolling volatility
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)

        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)

        # Calculate consistency using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)

        cv = vol_rolling_std / (vol_rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)

        return consistency.fillna(0).values

    def _calculate_volatility_of_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility of volatility - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))

        # Vectorized volatility of volatility calculation
        returns_series = pd.Series(returns)

        # Calculate rolling volatility for both windows
        vol1 = self._vectorbt_rolling_operation(returns_series, "std", window).shift(window)
        vol2 = self._vectorbt_rolling_operation(returns_series, "std", window)

        # Volatility of volatility
        vol_of_vol = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)

        return vol_of_vol.values

    def _calculate_volatility_uncertainty(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime uncertainty - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))

        # Vectorized uncertainty calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)

        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)

        # Calculate uncertainty using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)

        vol_vol = vol_rolling_std / (vol_rolling_mean + 1e-8)
        uncertainty = vol_vol.clip(0, 1)

        return uncertainty.fillna(0).values

    def _detect_volatility_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect volatility regime changes - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))

        # Vectorized regime change detection
        returns_series = pd.Series(returns)

        # Calculate rolling volatility for both windows
        vol1 = self._vectorbt_rolling_operation(returns_series, "std", window).shift(window)
        vol2 = self._vectorbt_rolling_operation(returns_series, "std", window)

        # Significant change threshold (50% change)
        change_ratio = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)
        changes = (change_ratio > 0.5).astype(int)

        return changes.values

    def _calculate_volatility_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime transition probability - OPTIMIZED VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))

        # OPTIMIZED: Use vectorized transition probability calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 2)

        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)

        # Vectorized transition probability using volatility changes
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(rolling_vol)
        vol_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)

        # Transition probability based on volatility change rate
        transition_prob = vol_changes.abs() / (vol_mean + 1e-8)
        transition_prob = transition_prob.clip(0, 1)

        return transition_prob.fillna(0).values

    def _calculate_volatility_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime stability - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))

        # Vectorized stability calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)

        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)

        # Calculate stability using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)

        vol_vol = vol_rolling_std / (vol_rolling_mean + 1e-8)
        stability = (1 - vol_vol).clip(0, 1)

        return stability.fillna(0).values

    def _calculate_regime_persistence_score(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime persistence score - OPTIMIZED VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))

        # OPTIMIZED: Use VectorBT rolling operations for better performance
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)

        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)

        # Use VectorBT rolling apply for autocorrelation calculation
        if self.vectorbt_optimizer:
            try:
                persistence = self.vectorbt_rolling_optimizer.rolling_apply(
                    rolling_vol,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
window
                ).fillna(0)
            except Exception as e:
                tprint(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                persistence = rolling_vol.rolling(window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                ).fillna(0)
        else:
            # Fallback to pandas
            persistence = rolling_vol.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            ).fillna(0)

        return persistence.values

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if self.vectorbt_optimizer:
            try:
                if operation == 'mean':
                    return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    if func is not None:
                        return self.vectorbt_rolling_optimizer.rolling_apply(data, func, window, **kwargs)
                    else:
                        raise ValueError("Function must be provided for rolling apply operation")
                elif operation == 'corr':
                    other = kwargs.get('other')
                    if other is not None:
                        return self.vectorbt_rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                    else:
                        raise ValueError("Other series must be provided for rolling correlation")
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.vectorbt_rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
                elif operation == 'skew':
                    return self.vectorbt_rolling_optimizer.rolling_skew(data, window, **kwargs)
                elif operation == 'kurt':
                    return self.vectorbt_rolling_optimizer.rolling_kurt(data, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window, **kwargs)

        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'apply':
            func = kwargs.get('func')
            if func is not None:
                return rolling_obj.apply(func)
            else:
                raise ValueError("Function must be provided for rolling apply operation")
        elif operation == 'corr':
            other = kwargs.get('other')
            if other is not None:
                return rolling_obj.corr(other)
            else:
                raise ValueError("Other series must be provided for rolling correlation")
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

class RegimeVolumeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volume regime features optimized for 15m timeframe."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizers
        self.vectorbt_rolling_optimizer = None
        self.unified_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True
                )
                self.unified_optimizer = get_unified_optimization_system()
                tprint("✅ VectorBT optimizers initialized for RegimeVolumeFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_volume_features",
            category=FeatureCategory.VOLUME,
            description="Volume regime features for 15m timeframe regime classification",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 30, 80],  # 3h, 7.5h, 20h in 15m periods
                "persistence_windows": [8, 20, 64],  # 2h, 5h, 16h
                "clustering_windows": [16, 40, 128],  # 4h, 10h, 32h
                "transition_windows": [4, 12, 32]  # 1h, 3h, 8h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single volume regime feature as required by the base class."""
        try:
            # Generate all volume features
            features_dict = self.generate_features(data, **kwargs)

            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple volume feature if no features generated
                if 'volume' in data.columns and len(data) > 0:
                    volume_feature = data['volume'].pct_change().fillna(0).values
                    return pd.Series(volume_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            tprint(f"_generate_feature: Volume feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate volume regime features."""
        features = {}

        # Validate volume data
        if 'volume' not in data.columns:
            return features

        volume = data['volume'].values
        if len(volume) < 4:
            return features

        # 1. Volume Regime Persistence
        features.update(self._generate_volume_persistence_features(volume, data))

        # 2. Volume Clustering Features
        features.update(self._generate_volume_clustering_features(volume, data))

        # 3. Volume-Price Relationship Features
        features.update(self._generate_volume_price_features(volume, data))

        # 4. Volume Regime Transitions
        features.update(self._generate_volume_transition_features(volume, data))

        # 5. Volume Regime Stability
        features.update(self._generate_volume_stability_features(volume, data))

        return features

    def _generate_volume_persistence_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume persistence features for regime detection."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        for window in windows:
            if len(volume) < window:
                continue

            # Rolling volume statistics
            vol_mean = self._rolling_mean(volume, window)
            vol_std = self._rolling_std(volume, window)

            # Volume persistence (autocorrelation of volume)
            vol_persistence = self._calculate_volume_persistence(volume, window)

            # Volume regime strength
            vol_regime_strength = self._calculate_volume_regime_strength(volume, window)

            # Volume regime consistency
            vol_consistency = self._calculate_volume_consistency(volume, window)

            # Pad to match data length
            vol_mean_padded = np.full(len(data), np.nan)
            vol_std_padded = np.full(len(data), np.nan)
            vol_persistence_padded = np.full(len(data), np.nan)
            vol_strength_padded = np.full(len(data), np.nan)
            vol_consistency_padded = np.full(len(data), np.nan)

            # Rolling functions return len(volume) - window + 1, aligned to volume[window-1:]
            vol_mean_padded[window-1:] = vol_mean
            vol_std_padded[window-1:] = vol_std
            # Full-length functions return len(volume) with valid values from index window onwards
            vol_persistence_padded[window:] = vol_persistence[window:]
            vol_strength_padded[window:] = vol_regime_strength[window:]
            vol_consistency_padded[window:] = vol_consistency[window:]

            features[f'vol_mean_{window}'] = vol_mean_padded
            features[f'vol_std_{window}'] = vol_std_padded
            features[f'vol_persistence_{window}'] = vol_persistence_padded
            features[f'vol_regime_strength_{window}'] = vol_strength_padded
            features[f'vol_consistency_{window}'] = vol_consistency_padded

        return features

    def _generate_volume_clustering_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume clustering features."""
        features = {}
        windows = self.config.parameters["clustering_windows"]

        for window in windows:
            if len(volume) < window:
                continue

            # Volume clustering (similar to volatility clustering)
            vol_clustering = self._calculate_volume_clustering(volume, window)

            # Volume regime patterns
            vol_patterns = self._calculate_volume_patterns(volume, window)

            # Volume regime intensity
            vol_intensity = self._calculate_volume_intensity(volume, window)

            # Pad to match data length
            clustering_padded = np.full(len(data), np.nan)
            patterns_padded = np.full(len(data), np.nan)
            intensity_padded = np.full(len(data), np.nan)

            # Functions return len(volume) with valid values from index window onwards
            clustering_padded[window:] = vol_clustering[window:]
            patterns_padded[window:] = vol_patterns[window:]
            intensity_padded[window:] = vol_intensity[window:]

            features[f'vol_clustering_{window}'] = clustering_padded
            features[f'vol_patterns_{window}'] = patterns_padded
            features[f'vol_intensity_{window}'] = intensity_padded

        return features

    def _generate_volume_price_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume-price relationship features."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        if 'close' not in data.columns:
            return features

        close_prices = data['close'].values
        if len(close_prices) != len(volume):
            # Log the mismatch and return empty features - this shouldn't happen
            tprint(f"⚠️ WARNING: Length mismatch detected - volume={len(volume)}, close_prices={len(close_prices)}")
            tprint(f"⚠️ This indicates a data integrity issue. Skipping volume-price features.")
            return features

        for window in windows:
            if len(volume) < window:
                continue

            # Volume-price correlation
            vol_price_corr = self._calculate_volume_price_correlation(volume, close_prices, window)

            # Volume-weighted price change
            vol_weighted_price = self._calculate_volume_weighted_price_change(volume, close_prices, window)

            # Volume regime price impact
            vol_price_impact = self._calculate_volume_price_impact(volume, close_prices, window)

            # Pad to match data length
            corr_padded = np.full(len(data), np.nan)
            weighted_padded = np.full(len(data), np.nan)
            impact_padded = np.full(len(data), np.nan)

            # Functions return len(volume) with valid values from index window onwards
            corr_padded[window:] = vol_price_corr[window:]
            weighted_padded[window:] = vol_weighted_price[window:]
            impact_padded[window:] = vol_price_impact[window:]

            features[f'vol_price_corr_{window}'] = corr_padded
            features[f'vol_weighted_price_{window}'] = weighted_padded
            features[f'vol_price_impact_{window}'] = impact_padded

        return features

    def _generate_volume_transition_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(volume) < window * 2:
                continue

            # Volume regime change detection
            vol_change = self._detect_volume_regime_changes(volume, window)

            # Volume regime transition probability
            transition_prob = self._calculate_volume_transition_probability(volume, window)

            # Volume regime momentum
            vol_momentum = self._calculate_volume_momentum(volume, window)

            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)
            momentum_padded = np.full(len(data), np.nan)

            # Functions return len(volume) with valid values from index window*2 onwards
            change_padded[window*2:] = vol_change[window*2:]
            prob_padded[window*2:] = transition_prob[window*2:]
            momentum_padded[window*2:] = vol_momentum[window*2:]

            features[f'vol_regime_change_{window}'] = change_padded
            features[f'vol_transition_prob_{window}'] = prob_padded
            features[f'vol_momentum_{window}'] = momentum_padded

        return features

    def _generate_volume_stability_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume regime stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(volume) < window:
                continue

            # Volume regime stability
            vol_stability = self._calculate_volume_stability(volume, window)

            # Volume regime persistence score
            persistence_score = self._calculate_volume_persistence_score(volume, window)

            # Volume regime entropy
            vol_entropy = self._calculate_volume_entropy(volume, window)

            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)
            entropy_padded = np.full(len(data), np.nan)

            # These functions return len(volume) with valid values from index window onwards
            stability_padded[window:] = vol_stability[window:]
            persistence_padded[window:] = persistence_score[window:]
            entropy_padded[window:] = vol_entropy[window:]

            features[f'vol_stability_{window}'] = stability_padded
            features[f'vol_persistence_score_{window}'] = persistence_padded
            features[f'vol_entropy_{window}'] = entropy_padded

        return features

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean - VECTORIZED."""
        if len(data) < window:
            return np.array([])

        # Vectorized approach using pandas rolling
        data_series = pd.Series(data)
        result = self._vectorbt_rolling_operation(data_series, "mean", window).dropna().values

        return result

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation - VECTORIZED."""
        if len(data) < window:
            return np.array([])

        # Vectorized approach using pandas rolling
        data_series = pd.Series(data)
        result = self._vectorbt_rolling_operation(data_series, "std", window).dropna().values

        return result

    def _calculate_volume_persistence(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence using autocorrelation - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized autocorrelation calculation
        persistence = np.zeros(len(volume))

        # Calculate autocorrelation using vectorized operations
        vol_series = pd.Series(volume)

        # Vectorized autocorrelation using rolling correlation with shifted series
        vol_shifted = vol_series.shift(1)
        autocorr = vol_series.rolling(window).corr(vol_shifted).fillna(0)

        persistence[window:] = autocorr[window:]
        return persistence

    def _calculate_volume_regime_strength(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime strength - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # Vectorized regime strength calculation
        vol_series = pd.Series(volume)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)

        # Regime strength based on consistency of volume level
        vol_consistency = 1.0 - (rolling_std / (rolling_mean + 1e-8))
        strength = vol_consistency.clip(0, 1)

        return strength.fillna(0).values

    def _calculate_volume_consistency(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime consistency - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # Vectorized consistency calculation
        vol_series = pd.Series(volume)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)

        # Consistency based on low coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)

        return consistency.fillna(0).values

    def _calculate_volume_clustering(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume clustering - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized autocorrelation calculation
        vol_series = pd.Series(volume)

        # Vectorized autocorrelation using rolling correlation with shifted series
        vol_shifted = vol_series.shift(1)
        clustering = vol_series.rolling(window).corr(vol_shifted).fillna(0)

        return clustering.values

    def _calculate_volume_patterns(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime patterns - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized pattern calculation
        vol_series = pd.Series(volume)

        # Vectorized trend calculation using rolling linear regression
        # Create index arrays for linear regression
        x = np.arange(window)

        # Calculate rolling trend using vectorized operations
        patterns = np.full(len(volume), 0.5)  # Default value

        for i in range(window, len(volume) + 1):
            vol_window = volume[i-window:i]
            if len(vol_window) >= 3:
                # Calculate trend using linear regression
                trend = np.polyfit(x, vol_window, 1)[0]
                # Normalize trend to 0-1 range
                mean_vol = np.mean(vol_window)
                if mean_vol > 0:
                    normalized_trend = (np.tanh(trend / mean_vol) + 1) / 2
                    patterns[i-1] = normalized_trend

        return patterns

    def _calculate_volume_intensity(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime intensity - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # Vectorized intensity calculation
        vol_series = pd.Series(volume)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)

        # Intensity based on volume relative to historical average
        intensity = (vol_series / (rolling_mean + 1e-8)).clip(0, 2)

        return intensity.fillna(0).values

    def _calculate_volume_price_correlation(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-price correlation - VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))

        # Vectorized correlation calculation
        vol_series = pd.Series(volume)
        price_series = pd.Series(prices)

        # Fix: corr() method expects a Series, not a Rolling object
        correlation = vol_series.rolling(window).corr(price_series).fillna(0)

        return correlation.values

    def _calculate_volume_weighted_price_change(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-weighted price change - OPTIMIZED VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        prices_series = pd.Series(prices)

        # Calculate price changes
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series)

        # Vectorized volume-weighted calculation
        # Calculate rolling volume sums and price change sums
        vol_sum = self._vectorbt_rolling_operation(volume_series, "sum", window)
        vol_price_sum = (volume_series * price_changes).rolling(window).sum()

        # Volume-weighted price change
        weighted_change = vol_price_sum / (vol_sum + 1e-8)

        return weighted_change.fillna(0).values

    def _calculate_volume_price_impact(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume price impact - OPTIMIZED VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        prices_series = pd.Series(prices)

        # Calculate price and volume changes
        from ...utils.error_handling import safe_diff
        price_changes = safe_diff(prices_series).abs()
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(volume_series)

        # Vectorized impact calculation
        # Calculate rolling correlation between volume changes and price changes
        impact = vol_changes.rolling(window).corr(price_changes).fillna(0)

        return impact.values

    def _detect_volume_regime_changes(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Detect volume regime changes - VECTORIZED."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))

        # Vectorized approach using pandas rolling
        volume_series = pd.Series(volume)

        # Calculate rolling means for both windows
        vol1 = self._vectorbt_rolling_operation(volume_series, "mean", window)
        vol2 = vol1.shift(-window)  # Second window

        # Calculate change ratios
        change_ratios = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)

        # Apply threshold (30% change)
        changes = (change_ratios > 0.3).astype(int)

        return changes.fillna(0).values

    def _calculate_volume_transition_probability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime transition probability - OPTIMIZED VECTORIZED."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)

        # Calculate rolling volume changes for transition probability
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(volume_series)
        vol_volatility = self._vectorbt_rolling_operation(vol_changes, "std", window)
        vol_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)

        # Vectorized transition probability
        transition_prob = vol_volatility / (vol_mean + 1e-8)
        transition_prob = transition_prob.clip(0, 1)

        return transition_prob.fillna(0).values

    def _calculate_volume_momentum(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume momentum - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)

        # Calculate volume changes for momentum
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(volume_series)
        vol_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)

        # Vectorized momentum calculation
        momentum = self._vectorbt_rolling_operation(vol_changes, "mean", window) / (vol_mean + 1e-8)

        return momentum.fillna(0).values

    def _calculate_volume_stability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime stability - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)

        # Calculate rolling statistics for stability
        rolling_std = self._vectorbt_rolling_operation(volume_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)

        # Vectorized stability calculation using coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)

        return stability.fillna(0).values

    def _calculate_volume_persistence_score(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence score - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)

        # Calculate volume changes for autocorrelation
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(volume_series)

        # Vectorized persistence using rolling correlation with shifted series
        vol_changes_shifted = vol_changes.shift(1)
        persistence = vol_changes.rolling(window).corr(vol_changes_shifted).fillna(0)

        return persistence.values

    def _calculate_volume_entropy(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime entropy - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))

        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)

        # Calculate volume changes for entropy
        from ...utils.error_handling import safe_diff
        vol_changes = safe_diff(volume_series)

        # Vectorized entropy using rolling coefficient of variation
        rolling_std = self._vectorbt_rolling_operation(vol_changes, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_changes, "mean", window).abs()

        # Entropy proxy using coefficient of variation
        entropy = rolling_std / (rolling_mean + 1e-8)

        return entropy.fillna(0).values

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if self.vectorbt_optimizer:
            try:
                if operation == 'mean':
                    return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window).mean()
        elif operation == 'std':
            return data.rolling(window).std()
        elif operation == 'var':
            return data.rolling(window).var()
        elif operation == 'min':
            return data.rolling(window).min()
        elif operation == 'max':
            return data.rolling(window).max()
        elif operation == 'sum':
            return data.rolling(window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

# Advanced Regime Features (Entropy, Complexity, Fractal Dimension, etc.)
class RegimeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for regime entropy features with VectorBT optimization."""

    def __init__(self, window: int = 10):
        config = FeatureConfig(
            name=f"regime_entropy_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Regime entropy over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if OPTIMIZATION_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_manager = get_unified_optimization_system()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime entropy using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]

        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)

        # Use VectorBT rolling apply for optimized entropy calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized entropy calculation
                entropy_series = self.vectorbt_rolling_optimizer.rolling_apply(
                    close,
                    self._calculate_shannon_entropy_vectorized,
window
                )
                return entropy_series
            except Exception as e:
                warnings.warn(f"VectorBT entropy calculation failed: {e}, using fallback")
                return self._calculate_entropy_fallback(close, window, data.index)
        else:
            return self._calculate_entropy_fallback(close, window, data.index)

    def _calculate_shannon_entropy_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Shannon entropy for a segment (vectorized)."""
        if len(segment) == 0:
            return np.nan

        # Calculate histogram with fixed bins for consistency
        hist, _ = np.histogram(segment, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) == 0:
            return 0.0

        # Calculate Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small epsilon to avoid log(0)
        return entropy

    def _calculate_entropy_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback entropy calculation using pandas rolling."""
        entropy_values = []
        for i in range(len(close)):
            if i < window - 1:
                entropy_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                entropy = self._calculate_shannon_entropy_vectorized(segment)
                entropy_values.append(entropy)

        return pd.Series(entropy_values, index=index)

class RegimeComplexityGenerator(VectorizedFeatureGenerator):
    """Generator for regime complexity features with VectorBT optimization."""

    def __init__(self, window: int = 5):
        config = FeatureConfig(
            name=f"regime_complexity_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Regime complexity over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if OPTIMIZATION_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_manager = get_unified_optimization_system()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime complexity using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]

        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)

        # Use VectorBT rolling apply for optimized complexity calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized complexity calculation
                complexity_series = self.vectorbt_rolling_optimizer.rolling_apply(
                    close,
                    self._calculate_sample_entropy_vectorized,
window
                )
                return complexity_series
            except Exception as e:
                warnings.warn(f"VectorBT complexity calculation failed: {e}, using fallback")
                return self._calculate_complexity_fallback(close, window, data.index)
        else:
            return self._calculate_complexity_fallback(close, window, data.index)

    def _calculate_sample_entropy_vectorized(self, segment: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy for a segment (vectorized)."""
        return self._sample_entropy(segment, m, r)

    def _calculate_complexity_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback complexity calculation using pandas rolling."""
        complexity_values = []
        for i in range(len(close)):
            if i < window - 1:
                complexity_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                complexity = self._sample_entropy(segment, m=2, r=0.2)
                complexity_values.append(complexity)

        return pd.Series(complexity_values, index=index)

    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        try:
            N = len(data)
            if N < m + 1:
                return 0.0

            # Normalize data
            data = (data - np.mean(data)) / np.std(data)

            # Create template vectors
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])

            # Calculate distances
            distances = []
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j:
                        dist = np.max(np.abs(patterns[i] - patterns[j]))
                        distances.append(dist)

            if not distances:
                return 0.0

            # Count matches
            r_threshold = r * np.std(data)
            matches = np.sum(np.array(distances) <= r_threshold)

            if matches == 0:
                return 0.0

            # Calculate sample entropy
            phi = matches / (N - m + 1)
            return -np.log(phi) if phi > 0 else 0.0

        except Exception:
            return 0.0

class RegimeFractalDimensionGenerator(VectorizedFeatureGenerator):
    """Generator for regime fractal dimension features with VectorBT optimization."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_fractal_dimension_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Regime fractal dimension over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if OPTIMIZATION_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_manager = get_unified_optimization_system()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime fractal dimension using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]

        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)

        # Use VectorBT rolling apply for optimized fractal dimension calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized fractal dimension calculation
                fractal_series = self.vectorbt_rolling_optimizer.rolling_apply(
                    close,
                    self._calculate_higuchi_fractal_dimension_vectorized,
window
                )
                return fractal_series
            except Exception as e:
                warnings.warn(f"VectorBT fractal dimension calculation failed: {e}, using fallback")
                return self._calculate_fractal_fallback(close, window, data.index)
        else:
            return self._calculate_fractal_fallback(close, window, data.index)

    def _calculate_higuchi_fractal_dimension_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension for a segment (vectorized)."""
        return self._higuchi_fractal_dimension(segment)

    def _calculate_fractal_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback fractal dimension calculation using pandas rolling."""
        fractal_values = []
        for i in range(len(close)):
            if i < window - 1:
                fractal_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                fractal_dim = self._higuchi_fractal_dimension(segment)
                fractal_values.append(fractal_dim)

        return pd.Series(fractal_values, index=index)

    def _higuchi_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension."""
        try:
            N = len(data)
            if N < 10:
                return 1.0

            # Normalize data
            data = (data - np.mean(data)) / np.std(data)

            # Calculate L(k) for different k values
            k_values = range(1, min(10, N//4))
            L_values = []

            for k in k_values:
                L_sum = 0
                for m in range(k):
                    L = 0
                    for i in range(1, (N - m) // k):
                        L += abs(data[m + i*k] - data[m + (i-1)*k])
                    L = L * (N - 1) / ((N - m) // k * k)
                    L_sum += L

                L_values.append(L_sum / k)

            if len(L_values) < 2:
                return 1.0

            # Calculate fractal dimension
            k_log = np.log(k_values)
            L_log = np.log(L_values)

            # Linear regression
            slope, _ = np.polyfit(k_log, L_log, 1)
            fractal_dim = -slope

            return max(1.0, min(2.0, fractal_dim))  # Bound between 1 and 2

        except Exception:
            return 1.0

class RegimeHurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for regime Hurst exponent features with VectorBT optimization."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_hurst_exponent_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Regime Hurst exponent over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if OPTIMIZATION_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_manager = get_unified_optimization_system()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime Hurst exponent using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]

        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)

        # Use VectorBT rolling apply for optimized Hurst exponent calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized Hurst exponent calculation
                hurst_series = self.vectorbt_rolling_optimizer.rolling_apply(
                    close,
                    self._calculate_hurst_exponent_vectorized,
window
                )
                return hurst_series
            except Exception as e:
                warnings.warn(f"VectorBT Hurst exponent calculation failed: {e}, using fallback")
                return self._calculate_hurst_fallback(close, window, data.index)
        else:
            return self._calculate_hurst_fallback(close, window, data.index)

    def _calculate_hurst_exponent_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Hurst exponent for a segment (vectorized)."""
        return self._calculate_hurst_exponent(segment)

    def _calculate_hurst_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback Hurst exponent calculation using pandas rolling."""
        hurst_values = []
        for i in range(len(close)):
            if i < window - 1:
                hurst_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                hurst = self._calculate_hurst_exponent(segment)
                hurst_values.append(hurst)

        return pd.Series(hurst_values, index=index)

    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            N = len(data)
            if N < 10:
                return 0.5

            # Calculate returns
            returns = np.diff(data)

            # R/S analysis
            n_values = [N//4, N//2, N]
            rs_values = []

            for n in n_values:
                if n < 5:
                    continue

                # Calculate R/S for this n
                rs_sum = 0
                for i in range(0, N - n, n):
                    segment = returns[i:i+n]
                    mean_segment = np.mean(segment)
                    deviations = segment - mean_segment
                    cumulative_deviations = np.cumsum(deviations)

                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    S = np.std(segment)

                    if S > 0:
                        rs_sum += R / S

                if rs_sum > 0:
                    rs_values.append(rs_sum / (N // n))

            if len(rs_values) < 2:
                return 0.5

            # Calculate Hurst exponent
            n_log = np.log(n_values[:len(rs_values)])
            rs_log = np.log(rs_values)

            # Linear regression
            slope, _ = np.polyfit(n_log, rs_log, 1)
            hurst = slope

            return max(0.0, min(1.0, hurst))  # Bound between 0 and 1

        except Exception:
            return 0.5

class RegimeMemoryStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for regime memory strength features with VectorBT optimization."""

    def __init__(self, window: int = 10):
        config = FeatureConfig(
            name=f"regime_memory_strength_{window}",
            category=FeatureCategory.ADVANCED_STATISTICAL,
            description=f"Regime memory strength over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if OPTIMIZATION_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_manager = get_unified_optimization_system()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime memory strength using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]

        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)

        # Use VectorBT rolling apply for optimized memory strength calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized memory strength calculation
                memory_series = self.vectorbt_rolling_optimizer.rolling_apply(
                    close,
                    self._calculate_memory_strength_vectorized,
window
                )
                return memory_series
            except Exception as e:
                warnings.warn(f"VectorBT memory strength calculation failed: {e}, using fallback")
                return self._calculate_memory_fallback(close, window, data.index)
        else:
            return self._calculate_memory_fallback(close, window, data.index)

    def _calculate_memory_strength_vectorized(self, segment: np.ndarray) -> float:
        """Calculate memory strength for a segment (vectorized)."""
        return self._calculate_memory_strength(segment)

    def _calculate_memory_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback memory strength calculation using pandas rolling."""
        memory_values = []
        for i in range(len(close)):
            if i < window - 1:
                memory_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                memory_strength = self._calculate_memory_strength(segment)
                memory_values.append(memory_strength)

        return pd.Series(memory_values, index=index)

    def _calculate_memory_strength(self, data: np.ndarray) -> float:
        """Calculate memory strength using autocorrelation."""
        try:
            N = len(data)
            if N < 5:
                return 0.0

            # Calculate autocorrelation for different lags
            autocorrs = []
            for lag in range(1, min(5, N//2)):
                if lag < N:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))

            if not autocorrs:
                return 0.0

            # Memory strength is the average autocorrelation
            memory_strength = np.mean(autocorrs)
            return max(0.0, min(1.0, memory_strength))

        except Exception:
            return 0.0

# Factory functions for creating regime feature generators
def create_regime_feature_generators() -> List[FeatureGenerator]:
    """Create all regime feature generators."""
    generators = []

    # Core regime generators
    generators.append(RegimeStatisticalFeatureGenerator())
    generators.append(RegimeStructuralTrendFeatureGenerator())
    generators.append(RegimeVolatilityFeatureGenerator())
    generators.append(RegimeVolumeFeatureGenerator())

    # Advanced regime generators
    for window in [10, 20]:
        generators.append(RegimeEntropyGenerator(window))

    for window in [5, 10]:
        generators.append(RegimeComplexityGenerator(window))

    for window in [20, 30]:
        generators.append(RegimeFractalDimensionGenerator(window))

    for window in [20, 30]:
        generators.append(RegimeHurstExponentGenerator(window))

    for window in [10, 20]:
        generators.append(RegimeMemoryStrengthGenerator(window))

    return generators

def create_default_regime_generators() -> List[FeatureGenerator]:
    """Create default regime feature generators."""
    return create_regime_feature_generators()


# Aliases for backward compatibility
RegimeFeatureGenerator = RegimeStatisticalFeatureGenerator
StatisticalRegimeFeatureGenerator = RegimeStatisticalFeatureGenerator
StructuralTrendRegimeFeatureGenerator = RegimeStructuralTrendFeatureGenerator
VolatilityRegimeFeatureGenerator = RegimeVolatilityFeatureGenerator
VolumeRegimeFeatureGenerator = RegimeVolumeFeatureGenerator
# AdvancedRegimeFeatureGenerator = RegimeFeatureIntegration  # Defined later in file

def create_regime_generators() -> List[FeatureGenerator]:
    """Create regime feature generators (alias for backward compatibility)."""
    return create_regime_feature_generators()

def create_advanced_regime_generators() -> List[FeatureGenerator]:
    """Create advanced regime feature generators with VectorBT optimization."""
    generators = []

    # Enhanced regime entropy features with multiple windows
    for window in [5, 10, 15, 20, 25, 30]:
        generators.append(RegimeEntropyGenerator(window))

    # Enhanced regime complexity features
    for window in [3, 5, 7, 10, 15]:
        generators.append(RegimeComplexityGenerator(window))

    # Enhanced regime fractal dimension features
    for window in [10, 15, 20, 25, 30, 40]:
        generators.append(RegimeFractalDimensionGenerator(window))

    # Enhanced regime Hurst exponent features
    for window in [10, 15, 20, 25, 30, 40]:
        generators.append(RegimeHurstExponentGenerator(window))

    # Enhanced regime memory strength features
    for window in [5, 8, 10, 12, 15, 20]:
        generators.append(RegimeMemoryStrengthGenerator(window))

    return generators

def process_regime_features_batch(data: pd.DataFrame,
                                generators: Optional[List[FeatureGenerator]] = None,
                                use_vectorbt: bool = True,
                                **kwargs) -> pd.DataFrame:
    """
    Process regime features in batch using VectorBT optimizations.

    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters

    Returns:
        DataFrame with generated regime features
    """
    if generators is None:
        generators = create_regime_feature_generators()

    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            unified_optimizer = get_unified_optimization_system()

            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result

        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_regime_features_sequential(data, generators, **kwargs)
    else:
        return _process_regime_features_sequential(data, generators, **kwargs)

def _process_regime_features_sequential(data: pd.DataFrame,
                                      generators: List[FeatureGenerator],
                                      **kwargs) -> pd.DataFrame:
    """Process regime features sequentially (fallback)."""
    results = []

    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)

# Regime Feature Integration Classes
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

# Import regime feature thresholds
try:
    from src.config.regime_feature_thresholds import get_regime_feature_thresholds
except ImportError:
    def get_regime_feature_thresholds():
        return {"quality_thresholds": {}}

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime-focused feature generation."""
    # Regime feature categories to include
    include_volatility_regime: bool = True
    include_volume_regime: bool = True
    include_structural_trend: bool = True
    include_statistical_regime: bool = True

    # Feature quality filters (moderately relaxed for regime signal)
    min_regime_persistence: Optional[float] = None
    max_feature_noise_ratio: Optional[float] = None
    min_temporal_stability: Optional[float] = None

    # Enhanced regime quality features
    include_regime_quality_metrics: bool = True
    include_economic_significance: bool = True
    include_trading_viability: bool = True

    # Performance optimizations
    enable_parallel_processing: bool = True
    enable_matrix_optimization: bool = True
    max_parallel_workers: int = 4

    # 15-minute timeframe optimization
    optimize_for_15m: bool = True
    trade_duration_minutes: Tuple[int, int] = (5, 30)

    # Feature selection
    max_features_per_category: int = 100  # Increased to allow more features per category
    total_max_features: int = 500  # Increased to accommodate all desired features
    enable_feature_selection: bool = False  # Disable to generate all features

    # Composite scoring weights (exposed for regime tuning)
    persistence_weight: float = 0.5
    noise_penalty_weight: float = 0.3
    stability_weight: float = 0.2

    # Intensity weighting controls
    persistence_scale: float = 0.5
    probability_scale: float = 0.75

    def __post_init__(self) -> None:
        thresholds = get_regime_feature_thresholds()
        quality_thresholds = thresholds.get("quality_thresholds", {})

        if self.min_regime_persistence is None:
            self.min_regime_persistence = quality_thresholds.get("min_regime_persistence", 0.2)

        if self.max_feature_noise_ratio is None:
            self.max_feature_noise_ratio = quality_thresholds.get("max_feature_noise_ratio", 1.2)

        if self.min_temporal_stability is None:
            self.min_temporal_stability = quality_thresholds.get("min_temporal_stability", 0.1)

class RegimeFeatureIntegration(VectorizedFeatureGenerator):
    """
    Unified regime feature generator that excludes trading features.

    This class provides a comprehensive interface for generating regime-focused
    features specifically designed for regime clustering. It integrates all
    regime-related feature generators while filtering out trading-relevant features.

    Key Features:
    - Unified regime feature generation from multiple sources
    - Trading feature exclusion for pure regime analysis
    - Regime-focused feature selection and quality filtering
    - 15-minute timeframe optimization
    - Parallel processing support for performance
    - VectorBT optimization for high-performance calculations

    Parameters:
    - config: RegimeFeatureConfig or FeatureConfig object
        - include_volatility_regime: Include volatility regime features (default: True)
        - include_volume_regime: Include volume regime features (default: True)
        - include_structural_trend: Include structural trend features (default: True)
        - include_statistical_regime: Include statistical regime features (default: True)
        - enable_parallel_processing: Enable parallel feature generation (default: True)
        - enable_matrix_optimization: Enable matrix operation optimization (default: True)
        - total_max_features: Maximum number of features to generate (default: 100)

    Returns:
    - Dict[str, np.ndarray]: Dictionary of regime features suitable for clustering

    Example:
        >>> config = RegimeFeatureConfig(total_max_features=50)
        >>> generator = RegimeFeatureIntegration(config)
        >>> features = generator.generate_features(data)
        >>> print(f"Generated {len(features)} regime features for clustering")
    """

    def __init__(self, config: Optional[Union[RegimeFeatureConfig, FeatureConfig]] = None):
        if config is None:
            config = RegimeFeatureConfig()
        elif isinstance(config, FeatureConfig) and not isinstance(config, RegimeFeatureConfig):
            # Convert FeatureConfig to RegimeFeatureConfig
            config = RegimeFeatureConfig(
                include_volatility_regime=True,
                include_volume_regime=True,
                include_structural_trend=True,
                include_statistical_regime=True,
                min_regime_persistence=0.7,
                max_feature_noise_ratio=0.3,
                min_temporal_stability=0.6,
                optimize_for_15m=True,
                trade_duration_minutes=(5, 30),
                max_features_per_category=30,
                total_max_features=100,
                enable_feature_selection=True,
                persistence_weight=0.5,
                noise_penalty_weight=0.3,
                stability_weight=0.2
            )

        self.regime_config = config
        self.config = config

        # Track the most recent selection metadata for downstream reporting
        self._latest_quality_stats: Dict[str, Dict[str, float]] = {}
        self._latest_selection_scores: Dict[str, float] = {}
        self._latest_category_counts: Dict[str, int] = {}
        self._latest_target_count: int = getattr(config, 'total_max_features', 100)
        self._latest_intensity_scalers: Dict[str, float] = {}

        # Initialize VectorBT optimizers
        self.vectorbt_rolling_optimizer = None
        self.unified_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'enable_gpu_acceleration', False),
                    enable_parallel=getattr(config, 'enable_parallel_processing', True)
                )
                self.unified_optimizer = get_unified_optimization_system()
                tprint("✅ VectorBT optimizers initialized successfully")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

        # Initialize regime-focused feature generators
        self.volatility_generator = RegimeVolatilityFeatureGenerator() if config.include_volatility_regime else None
        self.volume_generator = RegimeVolumeFeatureGenerator() if config.include_volume_regime else None
        self.structural_trend_generator = RegimeStructuralTrendFeatureGenerator() if config.include_structural_trend else None
        self.statistical_generator = RegimeStatisticalFeatureGenerator() if config.include_statistical_regime else None

        # Initialize base config
        base_config = FeatureConfig(
            name="regime_feature_integration",
            category=FeatureCategory.REGIME,
            description="Unified regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,
            min_lookback=8,
            max_lookback=128,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )

        super().__init__(base_config, enable_matrix_ops=True)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if self.vectorbt_optimizer:
            try:
                if operation == 'mean':
                    return self.vectorbt_rolling_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_rolling_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_rolling_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_rolling_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_rolling_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_rolling_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'corr':
                    other = kwargs.get('other')
                    if other is not None:
                        return self.vectorbt_rolling_optimizer.rolling_corr(data, other, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window).mean()
        elif operation == 'std':
            return data.rolling(window).std()
        elif operation == 'var':
            return data.rolling(window).var()
        elif operation == 'min':
            return data.rolling(window).min()
        elif operation == 'max':
            return data.rolling(window).max()
        elif operation == 'sum':
            return data.rolling(window).sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            if other is not None:
                return data.rolling(window).corr(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate unified regime features as a single feature series."""
        try:
            # Generate all regime features
            features_dict = self.generate_features(data, **kwargs)

            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple feature if no features generated
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            error_msg = f"Regime feature generation failed: {e}"
            tprint(error_msg)
            raise ValueError(error_msg) from e

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate unified regime features, excluding trading features."""
        start_time = time.time()
        features = {}
        feature_names = []

        try:
            # 🚀 OPTIMIZATION: Data preprocessing for matrix operations
            tprint(f"🚀 Starting optimized regime feature generation...")
            optimized_data = data

            # Check if matrix optimization is enabled (with fallback for old configs)
            enable_matrix_opt = getattr(self.regime_config, 'enable_matrix_optimization', True)
            if enable_matrix_opt:
                optimization_start = time.time()
                optimized_data = self._optimize_matrix_operations(data)
                optimization_time = time.time() - optimization_start
                tprint(f"⚡ Data optimization completed in {optimization_time:.2f}s")

            # Prepare generators for execution
            generators = []
            if self.volatility_generator:
                generators.append(("volatility", self.volatility_generator))
            if self.volume_generator:
                generators.append(("volume", self.volume_generator))
            if self.structural_trend_generator:
                generators.append(("structural_trend", self.structural_trend_generator))
            if self.statistical_generator:
                generators.append(("statistical", self.statistical_generator))

            # Execute generators (parallel or sequential based on config)
            if generators:
                # Check if parallel processing is enabled (with fallback for old configs)
                enable_parallel = getattr(self.regime_config, 'enable_parallel_processing', True)
                if enable_parallel and len(generators) > 1:
                    parallel_results = self._parallel_feature_generation(generators, optimized_data, **kwargs)
                else:
                    # Sequential execution for debugging or single-threaded environments
                    parallel_results = self._sequential_feature_generation(generators, optimized_data, **kwargs)

                # Merge results
                for generator_name, generator_features in parallel_results.items():
                    if generator_features:
                        features.update(generator_features)
                        feature_names.extend(generator_features.keys())
                        tprint(f"✅ {generator_name}: {len(generator_features)} features")

            # Generate enhanced regime quality features (sequential - depends on other features)
            include_quality_metrics = getattr(self.regime_config, 'include_regime_quality_metrics', False)
            if include_quality_metrics:
                tprint(f"🔧 Generating regime quality metrics...")
                quality_start = time.time()
                quality_features = self._generate_regime_quality_features(optimized_data, **kwargs)
                quality_time = time.time() - quality_start
                tprint(f"Generated {len(quality_features)} quality features in {quality_time:.2f}s")
                features.update(quality_features)
                feature_names.extend(quality_features.keys())

            # OPTIMIZED: Apply quality filters only (no trading feature filter needed - all features are regime-focused)
            if getattr(self.regime_config, 'enable_feature_selection', True):
                tprint(f"🎯 STAGE 1: Starting feature selection pipeline")
                tprint(f"   📊 Input features: {len(features)} total features to evaluate")
                
                filter_start = time.time()
                tprint(f"🔍 STAGE 2: Applying quality filters...")
                filtered_features, quality_stats = self._apply_quality_filters(features, optimized_data)
                filter_time = time.time() - filter_start
                tprint(f"   ✅ Quality filtering completed in {filter_time:.2f}s")
                tprint(f"   📈 Features passed quality filters: {len(filtered_features)}/{len(features)} ({len(filtered_features)/len(features)*100:.1f}%)")

                tprint(f"⚖️ STAGE 3: Applying intensity weighting...")
                # Apply intensity weighting prior to feature selection
                filtered_features, intensity_scalers, quality_stats = self._apply_intensity_weighting(
                    filtered_features,
                    quality_stats
                )
                if intensity_scalers:
                    self._latest_intensity_scalers = intensity_scalers
                    tprint(f"   ✅ Intensity weighting applied to {len(intensity_scalers)} features")
                else:
                    self._latest_intensity_scalers = {
                        name: 1.0 for name in filtered_features.keys()
                    }
                    tprint(f"   ℹ️ No intensity weighting applied (using default 1.0)")

                # Ensure we keep exactly the configured number of features for optimal performance
                target_features = getattr(self.regime_config, 'total_max_features', 100)
                max_per_category = getattr(self.regime_config, 'max_features_per_category', 100)
                self._latest_target_count = target_features
                
                tprint(f"🎯 STAGE 4: Final feature selection")
                tprint(f"   📊 Target features: {target_features}")
                tprint(f"   📊 Max per category: {max_per_category}")
                tprint(f"   📊 Available features: {len(filtered_features)}")

                if len(filtered_features) > target_features:
                    tprint(f"🔍 Feature selection: {len(filtered_features)} → {target_features} features")
                    selection_start = time.time()
                    filtered_features, quality_stats = self._select_top_features(
                        filtered_features,
                        quality_stats,
                        target_features
                    )
                    selection_time = time.time() - selection_start
                    tprint(f"   ✅ Selection completed in {selection_time:.2f}s")
                elif len(filtered_features) < target_features:
                    tprint(f"⚠️ Only {len(filtered_features)} features available (target: {target_features})")
                    tprint(f"   💡 Consider relaxing quality filters or increasing generator count")
                else:
                    tprint(f"✅ Perfect: {len(filtered_features)} features (target: {target_features})")

                features = filtered_features
                tprint(f"🎉 STAGE 5: Feature selection completed!")
                tprint(f"   📊 Final features selected: {len(features)}")
                tprint(f"   📊 Target was: {target_features}")
                tprint(f"   📊 Success rate: {len(features)/target_features*100:.1f}% of target")
                
                # Persist the latest stats aligned with the selected features
                self._latest_quality_stats = {
                    name: quality_stats.get(name, {})
                    for name in features.keys()
                }
                self._latest_intensity_scalers = {
                    name: self._latest_quality_stats.get(name, {}).get('intensity_scaler', 1.0)
                    for name in features.keys()
                }
                if len(filtered_features) <= target_features:
                    persistence_weight = getattr(self.regime_config, 'persistence_weight', 0.5)
                    noise_penalty_weight = getattr(self.regime_config, 'noise_penalty_weight', 0.3)
                    stability_weight = getattr(self.regime_config, 'stability_weight', 0.2)
                    self._latest_selection_scores = {
                        name: (
                            persistence_weight * stats.get('persistence', 0.0)
                            - noise_penalty_weight * stats.get('noise_ratio', 0.0)
                            + stability_weight * stats.get('temporal_stability', 0.0)
                        ) if stats else 0.0
                        for name, stats in self._latest_quality_stats.items()
                    }
                    self._latest_category_counts = self._compute_category_counts(features.keys())

                filter_time = time.time() - filter_start
                tprint(f"Feature filtering and quality checks completed in {filter_time:.2f}s")

            total_time = time.time() - start_time
            tprint(f"🎯 Total regime feature generation completed in {total_time:.2f}s")
            return features

        except Exception as e:
            error_msg = f"Regime feature generation failed: {e}"
            tprint(error_msg)
            raise ValueError(error_msg) from e

    def _parallel_feature_generation(self, generators: List[Tuple[str, Any]], data: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute feature generators in parallel for maximum performance."""
        results = {}

        # OPTIMIZED: Determine optimal number of workers based on system resources
        max_workers_config = getattr(self.regime_config, 'max_parallel_workers', 4)
        # Use CPU count for optimal parallelization
        import os
        cpu_count = os.cpu_count() or 4
        max_workers = min(max_workers_config, len(generators), cpu_count)

        def generate_features_worker(generator_info):
            """Worker function for parallel feature generation."""
            name, generator = generator_info
            try:
                start_time = time.time()
                features = generator.generate_features(data, **kwargs)
                generation_time = time.time() - start_time
                tprint(f"⚡ {name}: {len(features) if features else 0} features in {generation_time:.2f}s")
                return name, features
            except Exception as e:
                tprint(f"❌ {name} generation failed: {e}")
                return name, {}

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(generate_features_worker, gen_info): gen_info[0]
                for gen_info in generators
            }

            # Collect results as they complete
            for future in as_completed(future_to_name):
                name, features = future.result()
                results[name] = features

        return results

    def _sequential_feature_generation(self, generators: List[Tuple[str, Any]], data: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute feature generators sequentially for debugging or single-threaded environments."""
        results = {}

        for name, generator in generators:
            try:
                start_time = time.time()
                features = generator.generate_features(data, **kwargs)
                generation_time = time.time() - start_time
                tprint(f"⚡ {name}: {len(features) if features else 0} features in {generation_time:.2f}s")
                results[name] = features
            except Exception as e:
                tprint(f"❌ {name} generation failed: {e}")
                results[name] = {}

        return results

    def _optimize_matrix_operations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for matrix operations by ensuring proper data types and alignment."""
        # Convert to optimal dtypes for matrix operations
        optimized_data = data.copy()

        # Count columns that need conversion
        numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
        conversion_count = 0

        # Ensure numeric columns are float32 for better memory usage and speed
        for col in numeric_columns:
            if optimized_data[col].dtype != np.float32:
                optimized_data[col] = optimized_data[col].astype(np.float32)
                conversion_count += 1

        # Log optimization details
        if conversion_count > 0:
            tprint(f"⚡ Converted {conversion_count} columns to float32 for matrix optimization")
        else:
            tprint(f"⚡ Data already optimized (all numeric columns are float32)")

        # Ensure data is aligned and contiguous for matrix operations
        optimized_data = optimized_data.copy()  # Force contiguous memory layout

        return optimized_data

    @lru_cache(maxsize=128)
    def _cached_data_hash(self, data_hash: str) -> str:
        """Cache data hash for repeated operations."""
        return data_hash

    def _filter_trading_features(self, features: Dict[str, np.ndarray], feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Filter out any remaining trading-relevant features."""
        trading_patterns = [
            'rsi', 'macd', 'stochastic', 'williams', 'momentum',
            'oscillator', 'signal', 'crossover', 'divergence',
            'candlestick', 'pattern', 'breakout', 'support', 'resistance',
            'bollinger', 'atr', 'cci', 'roc', 'mfi', 'obv', 'ema', 'sma'
        ]

        # OPTIMIZED: Use dictionary comprehension for faster filtering
        regime_patterns = {
            'volatility', 'volume_regime', 'trend_persistence',
            'regime_stability', 'correlation', 'distribution',
            'clustering', 'persistence', 'structural', 'statistical',
            'vol_persistence', 'vol_clustering', 'vol_stability',
            'vol_regime', 'trend_strength', 'market_structure'
        }

        filtered_features = {
            name: feature_array for name, feature_array in features.items()
            if not any(pattern in name.lower() for pattern in trading_patterns)
            and any(pattern in name.lower() for pattern in regime_patterns)
        }

        return filtered_features

    def _apply_quality_filters(self, features: Dict[str, np.ndarray], data: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """Apply quality filters and compute per-feature quality statistics."""
        tprint(f"🔍 Quality Filtering Details:")
        tprint(f"   📊 Total input features: {len(features)}")
        
        filtered_features: Dict[str, np.ndarray] = {}
        quality_stats: Dict[str, Dict[str, float]] = {}

        # Relaxed thresholds for statistical features
        statistical_patterns = ['statistical', 'distribution', 'returns_', 'skewness', 'kurtosis', 'autocorr', 'entropy']

        # OPTIMIZED: Use vectorized filtering with batch processing
        valid_features = {
            name: feature_array for name, feature_array in features.items()
            if feature_array is not None and len(feature_array) > 0
        }
        tprint(f"   ✅ Valid features (non-null, non-empty): {len(valid_features)}")

        # Batch process features by type for efficiency
        statistical_features = {
            name: feature_array for name, feature_array in valid_features.items()
            if any(pattern in name.lower() for pattern in statistical_patterns)
        }
        tprint(f"   📈 Statistical features: {len(statistical_features)}")

        other_features = {
            name: feature_array for name, feature_array in valid_features.items()
            if not any(pattern in name.lower() for pattern in statistical_patterns)
        }
        tprint(f"   🔧 Other features: {len(other_features)}")

        # Process statistical features with relaxed criteria
        tprint(f"   🔍 Processing {len(statistical_features)} statistical features (relaxed criteria)...")
        statistical_passed = 0
        for name, feature_array in statistical_features.items():
            passed, metrics = self._is_high_quality_regime_feature(feature_array, relaxed=True)
            if passed:
                filtered_features[name] = feature_array
                statistical_passed += 1
                if metrics:
                    quality_stats[name] = metrics
        tprint(f"   ✅ Statistical features passed: {statistical_passed}/{len(statistical_features)} ({statistical_passed/len(statistical_features)*100:.1f}%)")

        # Process other features with standard criteria
        tprint(f"   🔍 Processing {len(other_features)} other features (standard criteria)...")
        other_passed = 0
        for name, feature_array in other_features.items():
            passed, metrics = self._is_high_quality_regime_feature(feature_array)
            if passed:
                filtered_features[name] = feature_array
                other_passed += 1
                if metrics:
                    quality_stats[name] = metrics
        tprint(f"   ✅ Other features passed: {other_passed}/{len(other_features)} ({other_passed/len(other_features)*100:.1f}%)")

        tprint(f"📊 Quality filter results: {len(filtered_features)}/{len(features)} features passed")
        if quality_stats:
            avg_persistence = np.mean([m['persistence'] for m in quality_stats.values()])
            avg_noise = np.mean([m['noise_ratio'] for m in quality_stats.values()])
            avg_stability = np.mean([m['temporal_stability'] for m in quality_stats.values()])
            tprint(
                "   ➤ Avg quality metrics — "
                f"persistence: {avg_persistence:.3f}, "
                f"noise: {avg_noise:.3f}, "
                f"stability: {avg_stability:.3f}"
            )

        self._latest_quality_stats = quality_stats
        return filtered_features, quality_stats

    def _apply_intensity_weighting(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Scale features using persistence and probability based intensity multipliers."""

        if not features:
            return features, {}, quality_stats

        persistence_scale = getattr(self.regime_config, 'persistence_scale', 0.0)
        probability_scale = getattr(self.regime_config, 'probability_scale', 0.0)

        updated_features: Dict[str, np.ndarray] = {}
        intensity_scalers: Dict[str, float] = {}
        updated_quality_stats: Dict[str, Dict[str, float]] = dict(quality_stats)

        for name, feature_array in features.items():
            metrics = dict(quality_stats.get(name, {}))

            persistence = float(metrics.get('persistence', 0.0) or 0.0)
            persistence = max(persistence, 0.0)
            scale = 1.0 + (persistence_scale * persistence if persistence_scale else 0.0)

            probability_value = metrics.get('probability')
            if probability_value is None and probability_scale and feature_array is not None and 'prob' in name.lower():
                valid_values = feature_array[~np.isnan(feature_array)]
                if len(valid_values) > 0:
                    probability_value = float(np.clip(np.nanmean(valid_values), 0.0, 1.0))

            if probability_value is not None:
                probability_value = float(np.clip(probability_value, 0.0, 1.0))
                probability_boost = max(probability_value - 0.5, 0.0)
                scale *= 1.0 + (probability_scale * probability_boost if probability_scale else 0.0)
                metrics['probability'] = probability_value

            if scale <= 0:
                scale = 1.0

            metrics['intensity_scaler'] = scale
            intensity_scalers[name] = scale
            updated_quality_stats[name] = metrics

            if feature_array is not None:
                updated_features[name] = np.asarray(feature_array) * scale
            else:
                updated_features[name] = feature_array

        return updated_features, intensity_scalers, updated_quality_stats

    def _determine_feature_category(self, feature_name: str) -> str:
        """Classify feature names into high-level regime categories."""
        name = feature_name.lower()

        if 'volatility' in name or 'vol_' in name:
            return 'volatility_regime'
        if 'volume' in name or 'liquidity' in name:
            return 'volume_regime'
        if 'trend' in name or 'structural' in name:
            return 'structural_trend'
        if 'statistical' in name or 'distribution' in name or 'entropy' in name:
            return 'statistical_regime'
        if 'economic' in name or 'macro' in name:
            return 'economic_quality'
        if 'trading' in name or 'position' in name:
            return 'trading_viability'
        if 'stability' in name or 'persistence' in name or 'consistency' in name or 'quality' in name:
            return 'regime_quality'

        return 'other'

    def _compute_category_counts(self, feature_names: List[str]) -> Dict[str, int]:
        """Compute category counts for reporting."""
        counts: Dict[str, int] = defaultdict(int)
        for name in feature_names:
            counts[self._determine_feature_category(name)] += 1
        return dict(counts)

    def _select_top_features(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Dict[str, Dict[str, float]],
        target_count: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """Select top features using composite scoring with category caps."""
        try:
            if not features:
                return features, quality_stats

            tprint(f"🎯 Top Feature Selection Details:")
            tprint(f"   📊 Input features: {len(features)}")
            tprint(f"   📊 Target count: {target_count}")

            persistence_weight = getattr(self.regime_config, 'persistence_weight', 0.5)
            noise_penalty_weight = getattr(self.regime_config, 'noise_penalty_weight', 0.3)
            stability_weight = getattr(self.regime_config, 'stability_weight', 0.2)
            max_per_category = getattr(self.regime_config, 'max_features_per_category', target_count)
            
            tprint(f"   ⚖️ Scoring weights: persistence={persistence_weight}, noise_penalty={noise_penalty_weight}, stability={stability_weight}")
            tprint(f"   📊 Max per category: {max_per_category}")

            composite_scores: Dict[str, float] = {}
            variances: Dict[str, float] = {}

            for name, feature_array in features.items():
                valid_values = feature_array[~np.isnan(feature_array)] if feature_array is not None else np.array([])
                if len(valid_values) > 1:
                    variances[name] = float(np.var(valid_values))
                else:
                    variances[name] = 0.0

                metrics = quality_stats.get(name, {})
                composite_score = (
                    persistence_weight * metrics.get('persistence', 0.0)
                    - noise_penalty_weight * metrics.get('noise_ratio', 0.0)
                    + stability_weight * metrics.get('temporal_stability', 0.0)
                )

                # Fallback to variance if metrics are missing (e.g., relaxed filters)
                if not metrics:
                    composite_score += variances[name]

                composite_scores[name] = composite_score

            # Sort features by composite score then variance as tie-breaker
            sorted_feature_names = sorted(
                features.keys(),
                key=lambda n: (composite_scores.get(n, float('-inf')), variances.get(n, 0.0)),
                reverse=True
            )

            selected_features: Dict[str, np.ndarray] = {}
            selected_stats: Dict[str, Dict[str, float]] = {}
            category_counts: Dict[str, int] = defaultdict(int)
            categories_capped: List[str] = []

            for name in sorted_feature_names:
                if len(selected_features) >= target_count:
                    break

                category = self._determine_feature_category(name)
                if category_counts[category] >= max_per_category:
                    if category not in categories_capped:
                        categories_capped.append(category)
                    continue

                selected_features[name] = features[name]
                if name in quality_stats:
                    selected_stats[name] = quality_stats[name]
                category_counts[category] += 1

            if len(selected_features) < target_count:
                tprint(
                    f"⚠️ Category caps limited selection to {len(selected_features)}/{target_count} features."
                )

            # Log selection summary for verification
            tprint(
                "🎯 Composite feature selection completed: "
                f"{len(selected_features)}/{target_count} features retained"
            )
            tprint(
                "   ➤ Weights — "
                f"persistence: {persistence_weight:.2f}, "
                f"noise penalty: {noise_penalty_weight:.2f}, "
                f"stability: {stability_weight:.2f}"
            )
            if categories_capped:
                tprint(f"   ➤ Category caps reached for: {', '.join(categories_capped)}")
            
            # Show category breakdown
            tprint(f"   📊 Category breakdown:")
            for category, count in sorted(category_counts.items()):
                tprint(f"      • {category}: {count} features")

            preview_count = min(5, len(selected_features))
            if preview_count:
                top_preview = list(selected_features.keys())[:preview_count]
                tprint("   ➤ Top features by composite score:")
                for feature_name in top_preview:
                    tprint(
                        f"      • {feature_name}: "
                        f"score={composite_scores.get(feature_name, 0.0):.4f}, "
                        f"variance={variances.get(feature_name, 0.0):.4f}"
                    )

            self._latest_selection_scores = {
                name: composite_scores.get(name, 0.0)
                for name in selected_features.keys()
            }
            self._latest_category_counts = dict(category_counts)

            return selected_features, selected_stats

        except Exception as e:
            tprint(f"⚠️ Feature selection failed: {e}, returning original features")
            return features, quality_stats

    def _generate_regime_quality_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime quality assessment features."""
        features = {}

        try:
            # Economic significance features
            if getattr(self.regime_config, 'include_economic_significance', False):
                features.update(self._generate_economic_significance_features(data))

            # Trading viability features
            if getattr(self.regime_config, 'include_trading_viability', False):
                features.update(self._generate_trading_viability_features(data))

            # Regime stability features
            features.update(self._generate_regime_stability_features(data))

        except Exception as e:
            tprint(f"⚠️ Regime quality feature generation failed: {e}")

        return features

    def _generate_economic_significance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate economic significance features for regime quality."""
        features = {}

        try:
            # Price impact significance
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                price_volatility = returns.rolling(20).std()
                price_impact = price_volatility / price_volatility.mean()
                features['economic_price_impact'] = price_impact.fillna(0).values

            # Volume significance
            if 'volume' in data.columns:
                volume_ma = data['volume'].rolling(20).mean()
                volume_significance = data['volume'] / volume_ma
                features['economic_volume_significance'] = volume_significance.fillna(1).values

            # Market efficiency
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                price_range = (data['high'] - data['low']) / data['close']
                efficiency = 1.0 / (1.0 + price_range.rolling(20).mean())
                features['economic_market_efficiency'] = efficiency.fillna(0.5).values

        except Exception as e:
            tprint(f"⚠️ Economic significance features failed: {e}")

        return features

    def _generate_trading_viability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trading viability features for regime quality."""
        features = {}

        try:
            # Trading frequency viability
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                trading_frequency = 1.0 / (1.0 + volatility)
                features['trading_frequency_viability'] = trading_frequency.fillna(0.5).values

            # Position duration viability - OPTIMIZED
            if 'close' in data.columns:
                # OPTIMIZED: Use vectorized trend strength calculation
                close_prices = data['close']
                from ...utils.error_handling import safe_diff
                price_changes = safe_diff(close_prices)

                # Vectorized trend strength using rolling slope approximation
                trend_strength = price_changes.rolling(20).mean().abs()
                position_duration = 1.0 / (1.0 + trend_strength)
                features['trading_position_duration'] = position_duration.fillna(0.5).values

            # Liquidity viability
            if 'volume' in data.columns and 'close' in data.columns:
                liquidity = data['volume'] * data['close']
                liquidity_viability = liquidity / liquidity.rolling(20).mean()
                features['trading_liquidity_viability'] = liquidity_viability.fillna(1).values

        except Exception as e:
            tprint(f"⚠️ Trading viability features failed: {e}")

        return features

    def _generate_regime_stability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime stability features for quality assessment."""
        features = {}

        try:
            # Regime persistence - OPTIMIZED
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                # OPTIMIZED: Use vectorized autocorrelation calculation
                autocorr = returns.rolling(20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                ).fillna(0)
                features['regime_persistence'] = autocorr.values

            # Regime transition stability
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                from ...utils.error_handling import safe_diff
                rolling_std = returns.rolling(5).std()
                regime_changes = (safe_diff(rolling_std) != 0).astype(int)
                stability = 1.0 - regime_changes.rolling(20).mean()
                features['regime_transition_stability'] = stability.fillna(0.5).values

            # Regime consistency
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                consistency = 1.0 / (1.0 + returns.rolling(20).std())
                features['regime_consistency'] = consistency.fillna(0.5).values

        except Exception as e:
            tprint(f"⚠️ Regime stability features failed: {e}")

        return features

    def _is_high_quality_regime_feature(self, feature_array: np.ndarray, relaxed: bool = False) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check if a feature meets quality standards and compute its quality metrics."""
        try:
            # Remove NaN values for analysis
            valid_values = feature_array[~np.isnan(feature_array)]

            if len(valid_values) < 5:
                return False, None

            # Test 1: Regime persistence (autocorrelation)
            if len(valid_values) > 1:
                corr = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
                regime_persistence = corr if not np.isnan(corr) else 0.0
            else:
                regime_persistence = 0.0

            # Test 2: Low noise-to-signal ratio
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            noise_ratio = std_val / (abs(mean_val) + 1e-8)

            # Test 3: Temporal stability
            if len(valid_values) > 5:
                window = min(5, len(valid_values) // 2)
                rolling_means = []
                for i in range(window, len(valid_values)):
                    rolling_means.append(np.mean(valid_values[i-window:i]))

                if len(rolling_means) > 1:
                    temporal_stability = 1.0 - (np.std(rolling_means) / (np.mean(np.abs(rolling_means)) + 1e-8))
                else:
                    temporal_stability = 0.0
            else:
                temporal_stability = 0.0

            metrics = {
                'persistence': regime_persistence,
                'noise_ratio': noise_ratio,
                'temporal_stability': temporal_stability,
                'valid_length': float(len(valid_values))
            }

            # Apply extremely lenient quality thresholds to preserve ~500 features
            # Regime features are expected to change with market regimes, so be very permissive
            if relaxed:
                # Extremely lenient thresholds for statistical features
                result = (regime_persistence > 0.01 and  # Extremely low bar for autocorrelation
                         noise_ratio < 10.0 and     # Allow very high variability for regime changes
                         temporal_stability > -2.0 and # Allow very negative stability (regime transitions)
                         len(valid_values) >= 2)    # Allow very short sequences
                tprint(f"   Statistical feature: persistence={regime_persistence:.3f}, noise={noise_ratio:.3f}, stability={temporal_stability:.3f}, valid_vals={len(valid_values)}, result={result}")
                return result, metrics
            else:
                # Extremely lenient thresholds for ALL regime features
                # Goal: Keep ~500 features instead of filtering to 103
                result = (regime_persistence > 0.01 and  # Extremely low bar for autocorrelation
                         noise_ratio < 8.0 and      # Allow very high noise for regime transitions
                         temporal_stability > -1.5 and # Allow very negative stability for regime changes
                         len(valid_values) >= 2)   # Allow very short sequences
                tprint(f"   Regime feature: persistence={regime_persistence:.3f}, noise={noise_ratio:.3f}, stability={temporal_stability:.3f}, valid_vals={len(valid_values)}, result={result}")
                return result, metrics

        except:
            return False, None

    def get_feature_summary(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Get summary of generated regime features including selection metadata."""
        stats = quality_stats or self._latest_quality_stats or {}
        total_features = len(features)
        category_counts = self._latest_category_counts or self._compute_category_counts(features.keys())
        max_per_category = getattr(self.regime_config, 'max_features_per_category', total_features or 1)

        if stats:
            avg_persistence = float(np.mean([m.get('persistence', 0.0) for m in stats.values()]))
            avg_noise = float(np.mean([m.get('noise_ratio', 0.0) for m in stats.values()]))
            avg_stability = float(np.mean([m.get('temporal_stability', 0.0) for m in stats.values()]))
        else:
            avg_persistence = 0.0
            avg_noise = 0.0
            avg_stability = 0.0

        selection_scores = self._latest_selection_scores or {}
        top_ranked = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        summary = {
            'total_features': total_features,
            'feature_categories': category_counts,
            'quality_metrics': {
                'avg_persistence': avg_persistence,
                'avg_noise_ratio': avg_noise,
                'avg_temporal_stability': avg_stability
            },
            'selection': {
                'target': self._latest_target_count,
                'weights': {
                    'persistence': getattr(self.regime_config, 'persistence_weight', 0.5),
                    'noise_penalty': getattr(self.regime_config, 'noise_penalty_weight', 0.3),
                    'stability': getattr(self.regime_config, 'stability_weight', 0.2)
                },
                'intensity_scalers': self._latest_intensity_scalers or {
                    name: stats.get(name, {}).get('intensity_scaler', 1.0)
                    for name in features.keys()
                },
                'category_quota': {
                    category: {
                        'count': count,
                        'max': max_per_category
                    }
                    for category, count in category_counts.items()
                },
                'composite_scores': selection_scores,
                'top_ranked_features': top_ranked
            }
        }

        return summary

# Analyst Features - Regime generators
class AnalystRegimeProbTrendingGenerator(VectorizedFeatureGenerator):
    """Generator for regime probability trending feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_regime_prob_trending",
            category=FeatureCategory.REGIME,
            description="Analyst probability of trending regime",
            required_columns=[],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime probability trending feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
            if current_regime == 'trending':
                prob_trending = 1.0
            elif current_regime == 'choppy':
                prob_trending = 0.0
            else:
                prob_trending = 0.5
        else:
            prob_trending = 0.5

        prob_trending_series = pd.Series([prob_trending] * len(data), index=data.index, name=self.config.name)
        return prob_trending_series

class AnalystRegimeProbChoppyGenerator(VectorizedFeatureGenerator):
    """Generator for regime probability choppy feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_regime_prob_choppy",
            category=FeatureCategory.REGIME,
            description="Analyst probability of choppy regime",
            required_columns=[],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime probability choppy feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
            if current_regime == 'choppy':
                prob_choppy = 1.0
            elif current_regime == 'trending':
                prob_choppy = 0.0
            else:
                prob_choppy = 0.5
        else:
            prob_choppy = 0.5

        prob_choppy_series = pd.Series([prob_choppy] * len(data), index=data.index, name=self.config.name)
        return prob_choppy_series

class AnalystRegimeStabilityGenerator(VectorizedFeatureGenerator):
    """Generator for regime stability feature."""

    def __init__(self, lookback: int = 50):
        config = FeatureConfig(
            name="analyst_regime_stability",
            category=FeatureCategory.REGIME,
            description="Analyst regime stability (1 - regime_entropy)",
            required_columns=[],
            default_lookback=lookback,
            min_lookback=20,
            max_lookback=200,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime stability feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            regime = regime_data['regime']

            # Shannon entropy calculation
            regime_counts = regime.value_counts()
            total_regimes = len(regime_counts)
            if total_regimes > 0:
                regime_probs = regime_counts / len(regime)
                entropy = -np.sum(regime_probs * np.log2(regime_probs.replace(0, 1)))
                max_entropy = np.log2(total_regimes) if total_regimes > 1 else 1
                stability = 1 - (entropy / max_entropy)
            else:
                stability = 0.5
        else:
            stability = 0.5

        stability_series = pd.Series([stability] * len(data), index=data.index, name=self.config.name)
        return stability_series


# Convenience function for easy integration
def generate_regime_features(data: pd.DataFrame,
                           config: Optional[RegimeFeatureConfig] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate regime-focused features for clustering.

    Args:
        data: Market data DataFrame with OHLCV columns
        config: Configuration for regime feature generation

    Returns:
        Tuple of (features_dict, summary_dict)
    """
    if config is None:
        config = RegimeFeatureConfig()

    generator = RegimeFeatureIntegration(config)
    features = generator.generate_features(data)
    summary = generator.get_feature_summary(features, generator._latest_quality_stats)

    return features, summary

__all__ = [
    'RegimeStatisticalFeatureGenerator',
    'RegimeStructuralTrendFeatureGenerator',
    'RegimeVolatilityFeatureGenerator',
    'RegimeVolumeFeatureGenerator',
    'RegimeEntropyGenerator',
    'RegimeComplexityGenerator',
    'RegimeFractalDimensionGenerator',
    'RegimeHurstExponentGenerator',
    'RegimeMemoryStrengthGenerator',
    'RegimeFeatureConfig',
    'RegimeFeatureIntegration',
    'AnalystRegimeProbTrendingGenerator',
    'AnalystRegimeProbChoppyGenerator',
    'AnalystRegimeStabilityGenerator',
    'generate_regime_features',
    'create_regime_feature_generators',
    'create_default_regime_generators',
    'create_advanced_regime_generators',
    'create_regime_generators',
    'process_regime_features_batch',
    # Aliases for backward compatibility
    'RegimeFeatureGenerator',
    'StatisticalRegimeFeatureGenerator',
    'StructuralTrendRegimeFeatureGenerator',
    'VolatilityRegimeFeatureGenerator',
    'VolumeRegimeFeatureGenerator'
    'AdvancedRegimeFeatureGenerator'
]

# Aliases for backward compatibility and external imports
RegimeFeatureGenerator = RegimeFeatureIntegration
AdvancedRegimeFeatureGenerator = RegimeFeatureIntegration
StatisticalRegimeFeatureGenerator = RegimeStatisticalFeatureGenerator
StructuralTrendRegimeFeatureGenerator = RegimeStructuralTrendFeatureGenerator
VolatilityRegimeFeatureGenerator = RegimeVolatilityFeatureGenerator
VolumeRegimeFeatureGenerator = RegimeVolumeFeatureGenerator
