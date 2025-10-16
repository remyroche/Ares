"""
import warnings
Cross Timeframe TA-Lib Integration for Short-Term Crypto Trading

This module integrates the Top 20 TA-Lib indicators with cross-timeframe analysis,
optimized for short-term high-leverage crypto trading strategies.

Key Features:
- Multi-timeframe TA-Lib indicator generation
- Cross-timeframe correlation analysis
- Optimized parameter selection for different timeframes
- High-leverage risk management integration
- Real-time feature computation for live trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from pathlib import Path
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import our enhanced feature generators and cross-timeframe analysis
try:
    from .feature_generators import (
        FEATURE_GENERATORS, get_feature_generator,
        create_apo_config, create_cmo_config, create_natr_config, create_pfe_config,
        create_t3_config, create_kama_config, create_mama_config, create_aroon_oscillator_config, create_ppo_config,
        create_beta_config, create_true_range_config, create_rocr_config, create_adxr_config, create_tema_config,
        create_cdl_engulfing_config, create_cdl_morning_star_config, create_cdl_evening_star_config,
        create_cdl_three_white_soldiers_config, create_cdl_harami_config
    )
    from .cross_timeframe_analysis_pipeline import (
        CrossTimeframeConfig, CrossTimeframeResult, CrossTimeframeAnalysisPipeline
    )
    from src.utils.common_operations import get_m1_gpu_manager
    from src.utils.math_validation import safe_divide, validate_finite

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

# CuPy imports for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Integration dependencies not available: {e}")
    INTEGRATION_AVAILABLE = False

@dataclass
class TALibCrossTimeframeConfig:
    """Configuration for TA-Lib cross-timeframe analysis."""
    # Core timeframes for crypto scalping
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '30m'])

    # Indicator phases - optimized for short-term trading
    phase_1_indicators: List[str] = field(default_factory=lambda: [
        'apo', 'cmo', 'ultimate_oscillator', 'natr', 'pfe'
    ])
    phase_2_indicators: List[str] = field(default_factory=lambda: [
        't3', 'kama', 'mama', 'aroon_oscillator', 'ppo'
    ])
    phase_3_indicators: List[str] = field(default_factory=lambda: [
        'beta', 'true_range', 'rocr', 'adxr', 'tema'
    ])
    phase_4_indicators: List[str] = field(default_factory=lambda: [
        'cdl_engulfing', 'cdl_morning_star', 'cdl_evening_star',
        'cdl_three_white_soldiers', 'cdl_harami'
    ])

    # Parameter optimization for different timeframes
    timeframe_parameters: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        '1m': {'lookback_base': 5, 'fast_period': 3, 'slow_period': 8},      # Ultra-short for scalping
        '5m': {'lookback_base': 12, 'fast_period': 8, 'slow_period': 21},    # Short-term momentum
        '15m': {'lookback_base': 20, 'fast_period': 12, 'slow_period': 26},  # Medium-term trends
        '30m': {'lookback_base': 30, 'fast_period': 12, 'slow_period': 26}   # Longer-term context
    })

    # Cross-timeframe analysis settings
    enable_cross_correlations: bool = True
    enable_timeframe_divergence: bool = True
    enable_momentum_spillover: bool = True

    # High-leverage risk management
    enable_volatility_scaling: bool = True
    max_leverage_multiplier: float = 10.0
    risk_adjustment_factor: float = 0.02  # 2% risk per trade

    # Performance optimization
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

@dataclass
class TALibCrossTimeframeResult:
    """Results from TA-Lib cross-timeframe analysis."""
    # Feature matrices for each timeframe
    timeframe_features: Dict[str, pd.DataFrame]

    # Cross-timeframe analysis results
    cross_correlations: Dict[str, pd.DataFrame]
    timeframe_divergence: Dict[str, pd.Series]
    momentum_spillover: Dict[str, pd.Series]

    # Risk management metrics
    volatility_adjustments: Dict[str, pd.Series]
    leverage_recommendations: Dict[str, pd.Series]

    # Performance metrics
    computation_time: float
    feature_count: int
    quality_metrics: Dict[str, Any]

class TALibCrossTimeframeIntegration:
    """
    Integration class for TA-Lib indicators with cross-timeframe analysis.

    Optimized for short-term high-leverage crypto trading strategies.
    """

    def __init__(self, config: Optional[TALibCrossTimeframeConfig] = None):
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Required integration dependencies not available")

        self.config = config or TALibCrossTimeframeConfig()
        self.logger = logger.getChild('TALibCrossTimeframeIntegration')

        # Initialize optimization components
        self._initialize_optimizers()

        self.logger.info("🚀 TA-Lib Cross-Timeframe Integration initialized")
        self._log_configuration()

    def _initialize_optimizers(self):
        """Initialize hardware and processing optimizers."""
        self.gpu_manager = get_m1_gpu_manager()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers) if self.config.enable_parallel_processing else None

    def _log_configuration(self):
        """Log current configuration."""
        total_indicators = (len(self.config.phase_1_indicators) + len(self.config.phase_2_indicators) +
                          len(self.config.phase_3_indicators) + len(self.config.phase_4_indicators))

        self.logger.info(f"📊 Configuration: {len(self.config.timeframes)} timeframes, {total_indicators} indicators")
        self.logger.info(f"⚡ Performance: GPU={self.gpu_manager and self.gpu_manager.mps_available}, Parallel={self.executor is not None}")

    async def analyze_crypto_timeframes(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str = "BTC/USDT"
    ) -> TALibCrossTimeframeResult:
        """
        Perform comprehensive TA-Lib cross-timeframe analysis for crypto trading.

        Args:
            data_dict: Dictionary of OHLCV data for different timeframes
            symbol: Trading symbol for logging

        Returns:
            TALibCrossTimeframeResult with all analysis results
        """
        start_time = time.time()
        self.logger.info(f"🔄 Starting TA-Lib cross-timeframe analysis for {symbol}")

        try:
            # Phase 1: Generate features for each timeframe
            timeframe_features = await self._generate_timeframe_features(data_dict)

            # Phase 2: Calculate cross-timeframe relationships
            cross_correlations = await self._calculate_cross_correlations(timeframe_features)

            # Phase 3: Detect timeframe divergences
            timeframe_divergence = await self._detect_timeframe_divergence(timeframe_features)

            # Phase 4: Analyze momentum spillover
            momentum_spillover = await self._analyze_momentum_spillover(timeframe_features)

            # Phase 5: Generate risk management metrics
            volatility_adjustments, leverage_recommendations = await self._generate_risk_metrics(
                timeframe_features, data_dict
            )

            # Calculate performance metrics
            computation_time = time.time() - start_time
            feature_count = sum(len(features.columns) for features in timeframe_features.values())

            quality_metrics = self._calculate_quality_metrics(timeframe_features)

            result = TALibCrossTimeframeResult(
                timeframe_features=timeframe_features,
                cross_correlations=cross_correlations,
                timeframe_divergence=timeframe_divergence,
                momentum_spillover=momentum_spillover,
                volatility_adjustments=volatility_adjustments,
                leverage_recommendations=leverage_recommendations,
                computation_time=computation_time,
                feature_count=feature_count,
                quality_metrics=quality_metrics
            )

            self.logger.info(f"✅ Analysis completed in {computation_time:.2f}s - {feature_count} features generated")
            return result

        except Exception as e:
            self.logger.error(f"❌ Cross-timeframe analysis failed: {e}")
            raise

    async def _generate_timeframe_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate TA-Lib features for each timeframe."""
        timeframe_features = {}

        for timeframe, data in data_dict.items():
            self.logger.info(f"🔧 Generating features for {timeframe} timeframe")

            # Get timeframe-specific parameters
            params = self.config.timeframe_parameters.get(timeframe, self.config.timeframe_parameters['1m'])

            # Generate features for each phase
            all_features = pd.DataFrame(index=data.index)

            # Phase 1: Core momentum indicators
            phase1_features = await self._generate_phase_features(data, self.config.phase_1_indicators, params, "Phase 1")
            all_features = pd.concat([all_features, phase1_features], axis=1)

            # Phase 2: Fast trend following
            phase2_features = await self._generate_phase_features(data, self.config.phase_2_indicators, params, "Phase 2")
            all_features = pd.concat([all_features, phase2_features], axis=1)

            # Phase 3: Risk management
            phase3_features = await self._generate_phase_features(data, self.config.phase_3_indicators, params, "Phase 3")
            all_features = pd.concat([all_features, phase3_features], axis=1)

            # Phase 4: Pattern recognition
            phase4_features = await self._generate_phase_features(data, self.config.phase_4_indicators, params, "Phase 4")
            all_features = pd.concat([all_features, phase4_features], axis=1)

            timeframe_features[timeframe] = all_features

        return timeframe_features

    async def _generate_phase_features(self, data: pd.DataFrame, indicators: List[str],
                                     params: Dict[str, Any], phase_name: str) -> pd.DataFrame:
        """Generate features for a specific phase."""
        features = pd.DataFrame(index=data.index)

        for indicator_name in indicators:
            try:
                generator = get_feature_generator(indicator_name)
                if generator:
                    # Generate indicator with timeframe-specific parameters
                    feature_series = await self._generate_single_indicator(
                        generator, data, indicator_name, params
                    )
                    features = pd.concat([features, feature_series], axis=1)
                else:
                    self.logger.warning(f"⚠️ Generator not found for {indicator_name}")
            except Exception as e:
                self.logger.debug(f"Failed to generate {indicator_name}: {e}")

        return features

    async def _generate_single_indicator(self, generator: Callable, data: pd.DataFrame,
                                       indicator_name: str, params: Dict[str, Any]) -> pd.Series:
        """Generate a single indicator with optimized parameters."""
        try:
            # Adjust parameters based on indicator type
            if indicator_name in ['apo', 'ppo']:
                feature = generator(data, fast_period=params.get('fast_period', 5),
                                  slow_period=params.get('slow_period', 13))
            elif indicator_name in ['t3']:
                feature = generator(data, lookback=params.get('lookback_base', 5))
            elif indicator_name in ['kama', 'mama']:
                feature = generator(data, lookback=params.get('lookback_base', 30))
            elif indicator_name in ['cmo', 'natr', 'pfe', 'aroon_oscillator', 'beta', 'true_range',
                                  'rocr', 'adxr', 'tema']:
                feature = generator(data, lookback=params.get('lookback_base', 14))
            elif indicator_name.startswith('cdl_'):
                feature = generator(data)  # Pattern indicators don't need parameters
            else:
                feature = generator(data)  # Default generation

            return feature

        except Exception as e:
            self.logger.error(f"❌ Failed to generate {indicator_name}: {e}")
            return pd.Series([0.0] * len(data), index=data.index, name=f'{indicator_name}_error')

    async def _calculate_cross_correlations(self, timeframe_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate cross-timeframe correlations for momentum analysis."""
        correlations = {}

        if not self.config.enable_cross_correlations:
            return correlations

        timeframes = list(timeframe_features.keys())

        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i+1:]:
                try:
                    # Calculate rolling correlations between timeframes
                    features1 = timeframe_features[tf1]
                    features2 = timeframe_features[tf2]

                    # Find common features
                    common_features = set(features1.columns) & set(features2.columns)

                    if common_features:
                        corr_matrix = pd.DataFrame(index=features1.index, columns=list(common_features))

                        for feature in common_features:
                            # Calculate rolling correlation
                            corr_series = features1[feature].rolling(window=20).corr(features2[feature])
                            corr_matrix[feature] = corr_series

                        correlations[f'{tf1}_{tf2}'] = corr_matrix
                        self.logger.debug(f"✅ Calculated correlations between {tf1} and {tf2}")

                except Exception as e:
                    self.logger.debug(f"Failed to calculate correlations between {tf1} and {tf2}: {e}")

        return correlations

    async def _detect_timeframe_divergence(self, timeframe_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Detect divergences between timeframes for trading signals."""
        divergences = {}

        if not self.config.enable_timeframe_divergence:
            return divergences

        # Focus on key momentum indicators
        momentum_indicators = ['apo', 'cmo', 'ppo', 't3', 'kama']

        for indicator in momentum_indicators:
            try:
                # Collect indicator values across timeframes
                indicator_data = {}

                for timeframe, features in timeframe_features.items():
                    if indicator in features.columns:
                        # Resample to common timeframe for comparison
                        indicator_data[timeframe] = features[indicator]

                if len(indicator_data) >= 2:
                    # Calculate divergence signal
                    divergence_signal = self._calculate_divergence_signal(indicator_data)
                    divergences[f'{indicator}_divergence'] = divergence_signal

            except Exception as e:
                self.logger.debug(f"Failed to detect divergence for {indicator}: {e}")

        return divergences

    def _calculate_divergence_signal(self, indicator_data: Dict[str, pd.Series]) -> pd.Series:
        """Calculate divergence signal from multiple timeframes."""
        # Simple divergence detection: when shorter timeframe moves opposite to longer timeframe
        timeframes = sorted(indicator_data.keys())

        if len(timeframes) >= 2:
            short_tf = timeframes[0]  # Shortest timeframe
            long_tf = timeframes[-1]  # Longest timeframe

            short_series = indicator_data[short_tf]
            long_series = indicator_data[long_tf]

            # Calculate rate of change for divergence detection
            short_roc = short_series.pct_change(5)
            long_roc = long_series.pct_change(5)

            # Divergence signal: when short-term moves up but long-term moves down (or vice versa)
            divergence = ((short_roc > 0) & (long_roc < 0)) | ((short_roc < 0) & (long_roc > 0))
            divergence_signal = divergence.astype(int) * 2 - 1  # Convert to +1/-1

            return divergence_signal

        return pd.Series([0] * len(next(iter(indicator_data.values()))),
                        index=next(iter(indicator_data.values())).index)

    async def _analyze_momentum_spillover(self, timeframe_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Analyze momentum spillover effects between timeframes."""
        spillover = {}

        if not self.config.enable_momentum_spillover:
            return spillover

        # Analyze momentum transmission from higher to lower timeframes
        momentum_indicators = ['apo', 'cmo', 'ppo', 't3']

        for indicator in momentum_indicators:
            try:
                # Calculate momentum spillover effect
                spillover_effect = self._calculate_spillover_effect(
                    timeframe_features, indicator
                )
                spillover[f'{indicator}_spillover'] = spillover_effect

            except Exception as e:
                self.logger.debug(f"Failed to calculate spillover for {indicator}: {e}")

        return spillover

    def _calculate_spillover_effect(self, timeframe_features: Dict[str, pd.DataFrame],
                                   indicator: str) -> pd.Series:
        """Calculate momentum spillover from higher to lower timeframes."""
        # Simplified spillover calculation
        # In practice, this would use more sophisticated econometric methods

        if indicator in timeframe_features.get('1m', pd.DataFrame()).columns:
            short_term = timeframe_features['1m'][indicator]

            # Calculate spillover as the difference between short-term momentum
            # and what would be expected from longer-term trends
            spillover = short_term - self._vectorbt_rolling_operation(short_term, "mean", 20)
            spillover = spillover / (self._vectorbt_rolling_operation(short_term, "std", 20) + 1e-8)

            return spillover

        return pd.Series([0.0] * len(next(iter(timeframe_features.values()))),
                        index=next(iter(timeframe_features.values())).index)

    async def _generate_risk_metrics(self, timeframe_features: Dict[str, pd.DataFrame],
                                   data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """Generate risk management metrics for high-leverage trading."""
        volatility_adjustments = {}
        leverage_recommendations = {}

        for timeframe, features in timeframe_features.items():
            try:
                # Calculate volatility-based position sizing
                vol_adjustment = self._calculate_volatility_adjustment(features, timeframe)
                volatility_adjustments[timeframe] = vol_adjustment

                # Generate leverage recommendations
                leverage_rec = self._calculate_leverage_recommendation(
                    vol_adjustment, data_dict[timeframe]
                )
                leverage_recommendations[timeframe] = leverage_rec

            except Exception as e:
                self.logger.debug(f"Failed to generate risk metrics for {timeframe}: {e}")

        return volatility_adjustments, leverage_recommendations

    def _calculate_volatility_adjustment(self, features: pd.DataFrame, timeframe: str) -> pd.Series:
        """Calculate volatility-based position adjustment factor."""
        try:
            # Use NATR (if available) or approximate volatility
            if 'natr' in features.columns:
                volatility = features['natr']
            elif 'true_range' in features.columns:
                volatility = features['true_range'].rolling(window=20).mean()
            else:
                # Fallback: approximate volatility from price changes
                volatility = features.filter(like='close').pct_change().std() * 100

            # Convert to adjustment factor (inverse relationship with volatility)
            vol_adjustment = 1.0 / (1.0 + volatility)

            # Normalize to reasonable range
            vol_adjustment = (vol_adjustment - vol_adjustment.min()) / (vol_adjustment.max() - vol_adjustment.min())
            vol_adjustment = vol_adjustment * 0.8 + 0.2  # Scale to 0.2-1.0 range

            return vol_adjustment

        except Exception as e:
            self.logger.error(f"Failed to calculate volatility adjustment: {e}")
            return pd.Series([0.5] * len(features), index=features.index)

    def _calculate_leverage_recommendation(self, vol_adjustment: pd.Series,
                                         price_data: pd.DataFrame) -> pd.Series:
        """Calculate recommended leverage based on volatility and market conditions."""
        try:
            # Base leverage on volatility adjustment and trend strength
            base_leverage = vol_adjustment * self.config.max_leverage_multiplier

            # Reduce leverage during high volatility periods
            high_volatility = vol_adjustment < 0.3
            base_leverage.loc[high_volatility] *= 0.5

            # Cap maximum leverage
            base_leverage = base_leverage.clip(upper=self.config.max_leverage_multiplier)

            # Apply risk adjustment factor
            risk_adjusted_leverage = base_leverage * (1.0 - self.config.risk_adjustment_factor)

            return risk_adjusted_leverage

        except Exception as e:
            self.logger.error(f"Failed to calculate leverage recommendation: {e}")
            return pd.Series([1.0] * len(vol_adjustment), index=vol_adjustment.index)

    def _calculate_quality_metrics(self, timeframe_features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate quality metrics for the generated features."""
        try:
            metrics = {
                'total_features': sum(len(features.columns) for features in timeframe_features.values()),
                'timeframes_covered': len(timeframe_features),
                'feature_completeness': {},
                'feature_stability': {}
            }

            # Calculate feature completeness
            for timeframe, features in timeframe_features.items():
                completeness = (features.notna().sum().sum() /
                              (features.shape[0] * features.shape[1]))
                metrics['feature_completeness'][timeframe] = completeness

            # Calculate feature stability (coefficient of variation)
            for timeframe, features in timeframe_features.items():
                numeric_features = features.select_dtypes(include=[np.number])
                if not numeric_features.empty:
                    stability = numeric_features.std() / (numeric_features.mean().abs() + 1e-8)
                    stability = stability.mean()
                    metrics['feature_stability'][timeframe] = stability

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate quality metrics: {e}")
            return {}

    async def get_real_time_features(self, current_data: Dict[str, pd.DataFrame],
                                   lookback_window: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Generate real-time features for live trading.

        Args:
            current_data: Current OHLCV data for each timeframe
            lookback_window: Lookback window for feature calculation

        Returns:
            Dictionary of real-time features for each timeframe
        """
        try:
            # Trim data to lookback window
            trimmed_data = {}
            for timeframe, data in current_data.items():
                trimmed_data[timeframe] = data.tail(lookback_window)

            # Generate features using existing analysis pipeline
            result = await self.analyze_crypto_timeframes(trimmed_data)

            # Return latest features for trading decisions
            latest_features = {}
            for timeframe, features in result.timeframe_features.items():
                latest_features[timeframe] = features.iloc[-1:]  # Get most recent row

            return latest_features

        except Exception as e:
            self.logger.error(f"Failed to generate real-time features: {e}")
            return {}

# Convenience functions for easy integration
def create_crypto_trading_integration(
    enable_gpu: bool = True,
    enable_parallel: bool = True,
    max_leverage: float = 10.0
) -> TALibCrossTimeframeIntegration:
    """
    Create a pre-configured integration optimized for crypto trading.

    Args:
        enable_gpu: Enable
        enable_parallel: Enable parallel processing
        max_leverage: Maximum leverage multiplier

    Returns:
        Configured TALibCrossTimeframeIntegration instance
    """
    config = TALibCrossTimeframeConfig(
        enable_gpu_acceleration=enable_gpu,
        enable_parallel_processing=enable_parallel,
        max_leverage_multiplier=max_leverage,
        # Optimized timeframes for crypto scalping
        timeframes=['1m', '5m', '15m', '30m'],
        # Focus on most important indicators for short-term trading
        phase_1_indicators=['apo', 'cmo', 'natr', 'pfe'],  # Core momentum
        phase_2_indicators=['t3', 'ppo', 'aroon_oscillator'],  # Fast trends
        phase_3_indicators=['beta', 'true_range', 'rocr'],  # Risk management
        phase_4_indicators=['cdl_engulfing', 'cdl_harami']  # Key patterns
    )

    return TALibCrossTimeframeIntegration(config)

# Example usage function
async def analyze_crypto_pair(pair_data: Dict[str, pd.DataFrame],
                            integration: Optional[TALibCrossTimeframeIntegration] = None) -> TALibCrossTimeframeResult:
    """
    Example function to analyze a crypto trading pair.

    Args:
        pair_data: Dictionary with OHLCV data for different timeframes
        integration: Pre-configured integration (creates default if None)

    Returns:
        Complete analysis results
    """
    if integration is None:
        integration = create_crypto_trading_integration()

    return await integration.analyze_crypto_timeframes(pair_data)

if __name__ == "__main__":
    print("🎯 TA-Lib Cross-Timeframe Integration for Crypto Trading")
    print("=" * 60)
    print("This module provides:")
    print("✅ Top 20 TA-Lib indicators optimized for short-term crypto trading")
    print("✅ Cross-timeframe correlation analysis")
    print("✅ High-leverage risk management")
    print("✅ Real-time feature generation")
    print("✅ Hardware-accelerated computation")
    print("\n🚀 Ready for high-frequency crypto trading!")
    print("\nExample usage:")
    print("  integration = create_crypto_trading_integration()")
    print("  result = await integration.analyze_crypto_timeframes(data_dict)")

class VectorBTHelper:
    """Helper class for VectorBT operations."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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
