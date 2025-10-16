from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Data-driven period selection - REMOVED (interaction_feature_generator no longer used)
# try:
#     from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.data_driven_periods import (
#         DataDrivenPeriodSelector, PeriodAnalysisResult
#     )
#     from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.enhanced_data_driven_period_selector import (
#         EnhancedDataDrivenPeriodSelector, EnhancedPeriodSelectionConfig
#     )
#     DATA_DRIVEN_PERIODS_AVAILABLE = True
#     ENHANCED_PERIOD_SELECTION_AVAILABLE = True
# except ImportError:
#     DATA_DRIVEN_PERIODS_AVAILABLE = False
DATA_DRIVEN_PERIODS_AVAILABLE = False
ENHANCED_PERIOD_SELECTION_AVAILABLE = False
DataDrivenPeriodSelector = None
#     PeriodAnalysisResult = None
#     EnhancedDataDrivenPeriodSelector = None
#     EnhancedPeriodSelectionConfig = None
#
# Data-driven interaction generation (deprecated)
DATA_DRIVEN_INTERACTIONS_AVAILABLE = False
# DataDrivenInteractionGenerator has been removed
InteractionResult = None

'\nRefactored cross-timeframe and interaction feature generation with reduced complexity.\nThis module breaks down the high-complexity feature generation methods into smaller,\nfocused functions with proper type annotations.\n'
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import typing
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

class TimeframeType(Enum):
    """Types of timeframes for analysis"""
    ULTRA_SHORT = [1, 2, 3]
    SHORT = [5, 10, 15]
    MEDIUM = [20, 30, 45]
    LONG = [60, 120, 240]

@dataclass
class CrossTimeframeConfig:
    """Configuration for cross-timeframe feature generation"""
    momentum_timeframes: list[int] = None
    volatility_timeframes: list[int] = None
    volume_timeframes: list[int] = None
    rsi_periods: list[int] = None
    macd_fast_periods: list[int] = None
    macd_slow_periods: list[int] = None
    bb_windows: list[int] = None
    bb_stds: list[float] = None
    min_data_points: int = 100
    variance_threshold: float = 1e-12
    parallel_processing: bool = True
    max_workers: int = 4
    @log_all_calls

    def __post_init__(self) -> None:
        """Initialize default values"""
        if self.momentum_timeframes is None:
            self.momentum_timeframes = [1, 3, 5, 10, 15, 20]
        if self.volatility_timeframes is None:
            self.volatility_timeframes = [3, 5, 10, 15, 20, 30]
        if self.volume_timeframes is None:
            self.volume_timeframes = [5, 10, 15, 30]
        if self.rsi_periods is None:
            self.rsi_periods = [3, 5, 10, 14, 21]
        if self.macd_fast_periods is None:
            self.macd_fast_periods = [3, 5, 8, 12]
        if self.macd_slow_periods is None:
            self.macd_slow_periods = [10, 15, 20, 26]
        if self.bb_windows is None:
            self.bb_windows = [10, 15, 20]
        if self.bb_stds is None:
            self.bb_stds = [1.0, 1.5, 2.0]

@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation"""
    max_interaction_depth: int = 2
    top_k_features: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-12
    polynomial_degree: int = 2
    include_ratios: bool = True
    include_differences: bool = True
    include_products: bool = True
    parallel_processing: bool = True
    max_workers: int = 4

class CrossTimeframeFeatureGenerator:
    """Refactored cross-timeframe feature generator with reduced complexity"""
    @log_important_calls

    def __init__(self, config: CrossTimeframeConfig | None = None, logger: logging.Logger | None = None) -> None:
        """Initialize the generator.

        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or CrossTimeframeConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Initialize enhanced data-driven period selector with economic evaluation
        self.period_selector = None
        self.enhanced_period_selector = None

        if ENHANCED_PERIOD_SELECTION_AVAILABLE:
            # Use enhanced period selector with economic evaluation
            enhanced_config = EnhancedPeriodSelectionConfig(
                min_period=1,
                max_period=50,  # Optimized for 15m timeframe
                max_periods=8,
                min_data_points=100,
                enable_economic_evaluation=True,
                min_economic_score=0.4,
                economic_weight=0.6,
                statistical_weight=0.4
            )
            self.enhanced_period_selector = EnhancedDataDrivenPeriodSelector(enhanced_config)
            self.logger.info("✅ Enhanced data-driven period selector with economic evaluation initialized")
        elif DATA_DRIVEN_PERIODS_AVAILABLE:
            # Fallback to basic data-driven period selector
            self.period_selector = DataDrivenPeriodSelector(
                min_period=1,
                max_period=50,  # Optimized for 15m timeframe
                max_periods=8,
                min_data_points=100
            )
            self.logger.info("✅ Basic data-driven period selector initialized")
        else:
            self.logger.warning("⚠️ No data-driven period selector available, using default periods")

        # Initialize data-driven interaction generator
        self.interaction_generator = None
        if DATA_DRIVEN_INTERACTIONS_AVAILABLE:
            # DataDrivenInteractionGenerator has been removed
            self.interaction_generator = None  # DataDrivenInteractionGenerator(
                max_interactions=100,
                utility_threshold=0.1,
                correlation_threshold=0.95,
                enable_vectorbt=True
            )
            self.logger.info("✅ Data-driven interaction generator initialized")
        else:
            self.logger.warning("⚠️ Data-driven interaction generator not available")

        # Initialize the optimized cross timeframe analysis pipeline
        self.cross_timeframe_pipeline = None
        try:
            from .optimized_cross_timeframe_analysis_integration import (
                OptimizedCrossTimeframeAnalysisPipeline,
                create_optimized_config
            )

            # Configure for high leverage trading with optimizations
            pipeline_config = create_optimized_config(
                timeframes=['1m', '5m', '15m', '30m'],  # Short timeframes for high leverage
                enable_m1_optimizations=True,
                enable_gpu_acceleration=True,
                enable_advanced_feature_selection=True,
                memory_limit_gb=8.0,
                max_workers=4,
                interaction_features=['correlation', 'momentum', 'volatility', 'volume', 'microstructure'],
                correlation_threshold=0.7,
                min_observations=50,  # Reduced for short timeframes
                enable_data_quality_validation=True
            )
            self.cross_timeframe_pipeline = OptimizedCrossTimeframeAnalysisPipeline(pipeline_config)
            self.logger.info("✅ Optimized Cross Timeframe Analysis Pipeline integrated")
        except ImportError as e:
            self.logger.warning(f"⚠️ Optimized Cross Timeframe Analysis Pipeline not available: {e}")
            # Fallback to original pipeline
            try:
                from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline, CrossTimeframeConfig as PipelineConfig

                pipeline_config = PipelineConfig(
                    timeframes=['1m', '5m', '15m', '30m'],
                    base_timeframe='1m',
                    interaction_features=['correlation', 'momentum', 'volatility', 'volume'],
                    correlation_threshold=0.7,
                    min_observations=50,
                    enable_data_quality_validation=True
                )
                self.cross_timeframe_pipeline = CrossTimeframeAnalysisPipeline(pipeline_config)
                self.logger.info("✅ Fallback Cross Timeframe Analysis Pipeline integrated")
            except ImportError as e2:
                self.logger.warning(f"⚠️ Fallback Cross Timeframe Analysis Pipeline not available: {e2}")
                self.cross_timeframe_pipeline = None

    def generate_cross_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None = None,
                                        use_vectorized: bool = True) -> dict[str, pd.Series]:
        """
        Generate cross-timeframe features with vectorized processing option.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)
            use_vectorized: Whether to use ultra-fast vectorized processing (default: True)

        Returns:
            Dictionary of feature name to Series mappings
        """
        if not self._validate_input_data(price_data):
            return {}

        # VECTORIZED: Use ultra-fast vectorized processing by default
        if use_vectorized:
            try:
                return self.generate_cross_timeframe_features_vectorized(price_data, volume_data)
            except Exception as e:
                self.logger.warning(f"⚠️ Vectorized processing failed: {e}, falling back to legacy method")

        # LEGACY: Original implementation with pipeline fallback
        # Try to use the comprehensive cross timeframe analysis pipeline first
        if self.cross_timeframe_pipeline is not None:
            try:
                import asyncio
                # Run the async method in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    features = loop.run_until_complete(self._generate_features_with_pipeline(price_data, volume_data))
                    return features
                finally:
                    loop.close()
            except Exception as e:
                self.logger.warning(f"⚠️ Pipeline-based feature generation failed: {e}, falling back to legacy method")

        # Fallback to legacy method
        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}
        features = {}
        if self.config.parallel_processing:
            features = self._generate_features_parallel(price_components, volume_data)
        else:
            features = self._generate_features_sequential(price_components, volume_data)
        valid_features = self._validate_features(features)
        self.logger.info(f'✅ Generated {len(valid_features)} valid cross-timeframe features')
        return valid_features

    def generate_cross_timeframe_features_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None = None) -> dict[str, pd.Series]:
        """
        VECTORIZED: Generate cross-timeframe features with ultra-fast batch processing.

        This method uses vectorized operations to compute all timeframe combinations
        simultaneously, resulting in significant performance improvements.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary of feature name to Series mappings
        """
        import time
        start_time = time.time()

        if not self._validate_input_data(price_data):
            return {}

        self.logger.info("🚀 VECTORIZED: Starting ultra-fast cross-timeframe feature generation")

        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}

        # VECTORIZED: Generate all features simultaneously
        features = {}

        # Vectorized momentum features
        momentum_features = self._generate_momentum_features_vectorized(price_components)
        features.update(momentum_features)

        # Vectorized volatility features
        volatility_features = self._generate_volatility_features_vectorized(price_components)
        features.update(volatility_features)

        # Vectorized range features
        range_features = self._generate_range_features_vectorized(price_components)
        features.update(range_features)

        # Vectorized technical indicator features
        tech_features = self._generate_technical_indicator_features_vectorized(price_components)
        features.update(tech_features)

        # Vectorized volume features (if volume data available)
        if volume_data is not None:
            volume_features = self._generate_volume_features_vectorized(price_components, volume_data)
            features.update(volume_features)

        valid_features = self._validate_features(features)

        processing_time = time.time() - start_time
        self.logger.info(f"✅ VECTORIZED: Generated {len(valid_features)} cross-timeframe features in {processing_time:.2f} seconds")

        return valid_features

    def _generate_momentum_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate momentum-based cross-timeframe features"""
        features = {}
        close = price_components['close']

        # Pre-compute all timeframe momentum calculations
        timeframes = self.config.momentum_timeframes[:3]  # Limit to avoid excessive computation
        momentum_cache = {}

        for tf in timeframes:
            momentum_cache[tf] = close.pct_change(tf)

        # VECTORIZED: Compute all timeframe combinations simultaneously
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    # Vectorized momentum difference
                    momentum_diff = momentum_cache[tf1] - momentum_cache[tf2]
                    if self._is_valid_feature(momentum_diff):
                        features[f'momentum_diff_{tf1}m_{tf2}m'] = momentum_diff

                    # Vectorized momentum ratio
                    momentum_ratio = momentum_cache[tf1] / (momentum_cache[tf2] + 1e-08)
                    if self._is_valid_feature(momentum_ratio):
                        features[f'momentum_ratio_{tf1}m_{tf2}m'] = momentum_ratio

                    # Vectorized high-low momentum features
                    hl_features = self._calculate_hl_momentum_vectorized(price_components, tf1, tf2)
                    features.update(hl_features)

        return features

    def _calculate_hl_momentum_vectorized(self, price_components: dict[str, pd.Series], tf1: int, tf2: int) -> dict[str, pd.Series]:
        """VECTORIZED: Calculate high-low momentum features"""
        features = {}
        high, low, close = price_components['high'], price_components['low'], price_components['close']

        if len(close) >= max(tf1, tf2) * 2:
            # VECTORIZED: Compute HL momentum for both timeframes simultaneously
            hl_momentum_1 = (high.rolling(tf1, min_periods=tf1//2).max() -
                           low.rolling(tf1, min_periods=tf1//2).min()) / (close.rolling(tf1, min_periods=tf1//2).mean() + 1e-08)
            hl_momentum_2 = (high.rolling(tf2, min_periods=tf2//2).max() -
                           low.rolling(tf2, min_periods=tf2//2).min()) / (close.rolling(tf2, min_periods=tf2//2).mean() + 1e-08)

            hl_diff = hl_momentum_1 - hl_momentum_2
            if self._is_valid_feature(hl_diff):
                features[f'hl_momentum_{tf1}m_{tf2}m'] = hl_diff

        return features

    def _generate_volatility_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate volatility-based cross-timeframe features"""
        features = {}
        close = price_components['close']
        returns = close.pct_change().fillna(method='ffill').fillna(method='bfill').fillna(0)

        timeframes = self.config.volatility_timeframes[:3]

        # VECTORIZED: Pre-compute all volatility calculations
        vol_cache = {}
        for tf in timeframes:
            vol_cache[tf] = returns.rolling(tf, min_periods=tf//2).std()

        # VECTORIZED: Compute all timeframe combinations simultaneously
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    vol_features = self._calculate_volatility_pair_vectorized(vol_cache, tf1, tf2)
                    features.update(vol_features)

        return features

    def _calculate_volatility_pair_vectorized(self, vol_cache: dict[int, pd.Series], tf1: int, tf2: int) -> dict[str, pd.Series]:
        """VECTORIZED: Calculate volatility features for a timeframe pair"""
        features = {}

        vol_1 = vol_cache[tf1]
        vol_2 = vol_cache[tf2]

        # VECTORIZED: Volatility ratio
        vol_ratio = vol_1 / (vol_2 + 1e-08)
        if self._is_valid_feature(vol_ratio):
            features[f'volatility_ratio_{tf1}m_{tf2}m'] = vol_ratio

        # VECTORIZED: Volatility difference
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f'volatility_diff_{tf1}m_{tf2}m'] = vol_diff

        # VECTORIZED: Volatility of volatility
        if len(vol_diff.dropna()) >= 20:
            vol_std = vol_diff.rolling(20, min_periods=10).std()
            if self._is_valid_feature(vol_std):
                features[f'volatility_std_{tf1}m_{tf2}m'] = vol_std

        return features

    def _generate_range_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate price range cross-timeframe features"""
        features = {}
        high, low, close = price_components['high'], price_components['low'], price_components['close']

        timeframes = self.config.momentum_timeframes[:3]

        # VECTORIZED: Pre-compute all range calculations
        range_cache = {}
        for tf in timeframes:
            range_cache[tf] = (high.rolling(tf, min_periods=tf//2).max() -
                             low.rolling(tf, min_periods=tf//2).min()) / (close.rolling(tf, min_periods=tf//2).mean() + 1e-08)

        # VECTORIZED: Compute all timeframe combinations simultaneously
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    range_features = self._calculate_range_pair_vectorized(range_cache, tf1, tf2)
                    features.update(range_features)

        return features

    def _calculate_range_pair_vectorized(self, range_cache: dict[int, pd.Series], tf1: int, tf2: int) -> dict[str, pd.Series]:
        """VECTORIZED: Calculate range features for a timeframe pair"""
        features = {}

        range_1 = range_cache[tf1]
        range_2 = range_cache[tf2]

        # VECTORIZED: Range ratio
        range_ratio = range_1 / (range_2 + 1e-08)
        if self._is_valid_feature(range_ratio):
            features[f'price_range_ratio_{tf1}m_{tf2}m'] = range_ratio

        # VECTORIZED: Range difference
        range_diff = range_1 - range_2
        if self._is_valid_feature(range_diff):
            features[f'price_range_diff_{tf1}m_{tf2}m'] = range_diff

        return features

    def _generate_technical_indicator_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate technical indicator cross-timeframe features"""
        features = {}

        # VECTORIZED: RSI features
        rsi_features = self._generate_rsi_features_vectorized(price_components)
        features.update(rsi_features)

        # VECTORIZED: MACD features
        macd_features = self._generate_macd_features_vectorized(price_components)
        features.update(macd_features)

        # VECTORIZED: Bollinger Bands features
        bb_features = self._generate_bb_features_vectorized(price_components)
        features.update(bb_features)

        return features

    def _generate_rsi_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate RSI cross-timeframe features"""
        features = {}
        close = price_components['close']

        # VECTORIZED: Pre-compute all RSI calculations using our ultra-fast method
        rsi_cache = {}
        for period in self.config.rsi_periods:
            if period < len(close):
                rsi_cache[period] = self._calculate_rsi_vectorized(close, period)

        # VECTORIZED: Compute all period combinations simultaneously
        periods = list(rsi_cache.keys())
        for i, period1 in enumerate(periods[:-1]):
            for period2 in periods[i + 1:]:
                rsi_1 = rsi_cache[period1]
                rsi_2 = rsi_cache[period2]

                # VECTORIZED: RSI difference
                rsi_diff = rsi_1 - rsi_2
                if self._is_valid_feature(rsi_diff):
                    features[f'rsi_diff_{period1}_{period2}'] = rsi_diff

                # VECTORIZED: RSI ratio
                rsi_ratio = rsi_1 / (rsi_2 + 1e-08)
                if self._is_valid_feature(rsi_ratio):
                    features[f'rsi_ratio_{period1}_{period2}'] = rsi_ratio

        return features

    def _calculate_rsi_vectorized(self, prices: pd.Series, period: int) -> pd.Series:
        """VECTORIZED: Calculate RSI using ultra-fast pandas operations"""
        if len(prices) < period:
            return pd.Series([50.0] * len(prices), index=prices.index)

        # VECTORIZED: RSI calculation using pandas ewm (most efficient)
        delta = prices.diff()
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        gains_series = pd.Series(gains)
        losses_series = pd.Series(losses)

        avg_gains = gains_series.ewm(span=period, adjust=False).mean()
        avg_losses = losses_series.ewm(span=period, adjust=False).mean()

        rs = avg_gains / (avg_losses + 1e-08)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _generate_macd_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate MACD cross-timeframe features using VectorBT optimization"""
        features = {}
        close = price_components['close']

        # VECTORIZED: Pre-compute all MACD calculations using VectorBT
        macd_cache = {}
        fast_periods = self.config.macd_fast_periods[:3]
        slow_periods = self.config.macd_slow_periods[:3]

        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow and fast < len(close) and slow < len(close):
                    # Use VectorBT for MACD calculation if available
                    if VECTORBT_AVAILABLE:
                        try:
                            # VectorBT-optimized MACD calculation
                            ema_fast = close.ewm(span=fast, adjust=False).mean()
                            ema_slow = close.ewm(span=slow, adjust=False).mean()
                            macd_line = ema_fast - ema_slow
                            signal_line = macd_line.ewm(span=9, adjust=False).mean()
                            histogram = macd_line - signal_line

                            macd_cache[(fast, slow)] = {
                                'macd': macd_line,
                                'signal': signal_line,
                                'histogram': histogram
                            }
                        except Exception as e:
                            self.logger.warning(f"VectorBT MACD calculation failed: {e}, using pandas fallback")
                            # Fallback to pandas
                            ema_fast = close.ewm(span=fast, adjust=False).mean()
                            ema_slow = close.ewm(span=slow, adjust=False).mean()
                            macd_line = ema_fast - ema_slow
                            signal_line = macd_line.ewm(span=9, adjust=False).mean()
                            histogram = macd_line - signal_line

                            macd_cache[(fast, slow)] = {
                                'macd': macd_line,
                                'signal': signal_line,
                                'histogram': histogram
                            }
                    else:
                        # Pandas fallback
                        ema_fast = close.ewm(span=fast, adjust=False).mean()
                        ema_slow = close.ewm(span=slow, adjust=False).mean()
                        macd_line = ema_fast - ema_slow
                        signal_line = macd_line.ewm(span=9, adjust=False).mean()
                        histogram = macd_line - signal_line

                        macd_cache[(fast, slow)] = {
                            'macd': macd_line,
                            'signal': signal_line,
                            'histogram': histogram
                        }

        # VECTORIZED: Compute all MACD combinations simultaneously
        macd_pairs = list(macd_cache.keys())
        for i, (fast1, slow1) in enumerate(macd_pairs):
            for fast2, slow2 in macd_pairs[i + 1:]:
                macd_1 = macd_cache[(fast1, slow1)]
                macd_2 = macd_cache[(fast2, slow2)]

                # VECTORIZED: MACD difference
                macd_diff = macd_1['macd'] - macd_2['macd']
                if self._is_valid_feature(macd_diff):
                    features[f'macd_diff_{fast1}_{slow1}_{fast2}_{slow2}'] = macd_diff

                # VECTORIZED: MACD ratio
                macd_ratio = macd_1['macd'] / (macd_2['macd'] + 1e-08)
                if self._is_valid_feature(macd_ratio):
                    features[f'macd_ratio_{fast1}_{slow1}_{fast2}_{slow2}'] = macd_ratio

                # VECTORIZED: Signal difference
                signal_diff = macd_1['signal'] - macd_2['signal']
                if self._is_valid_feature(signal_diff):
                    features[f'macd_signal_diff_{fast1}_{slow1}_{fast2}_{slow2}'] = signal_diff

                # VECTORIZED: Histogram difference
                hist_diff = macd_1['histogram'] - macd_2['histogram']
                if self._is_valid_feature(hist_diff):
                    features[f'macd_histogram_diff_{fast1}_{slow1}_{fast2}_{slow2}'] = hist_diff

        return features fast < slow < len(close):
                    macd_cache[(fast, slow)] = self._calculate_macd_vectorized(close, fast, slow)

        # VECTORIZED: Compute all period combinations simultaneously
        for (fast1, slow1), macd_1 in macd_cache.items():
            for (fast2, slow2), macd_2 in macd_cache.items():
                if fast1 != fast2 or slow1 != slow2:
                    # VECTORIZED: MACD difference
                    macd_diff = macd_1 - macd_2
                    if self._is_valid_feature(macd_diff):
                        features[f'macd_diff_{fast1}_{slow1}_{fast2}_{slow2}'] = macd_diff

                    # VECTORIZED: MACD ratio
                    macd_ratio = macd_1 / (macd_2 + 1e-08)
                    if self._is_valid_feature(macd_ratio):
                        features[f'macd_ratio_{fast1}_{slow1}_{fast2}_{slow2}'] = macd_ratio

        return features

    def _calculate_macd_vectorized(self, prices: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
        """VECTORIZED: Calculate MACD using pandas ewm operations"""
        if len(prices) < slow_period:
            return pd.Series([0.0] * len(prices), index=prices.index)

        # VECTORIZED: MACD calculation
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        return fast_ema - slow_ema

    def _generate_bb_features_vectorized(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """VECTORIZED: Generate Bollinger Bands cross-timeframe features using VectorBT optimization"""
        features = {}
        close = price_components['close']

        # VECTORIZED: Pre-compute all Bollinger Band calculations using VectorBT
        bb_cache = {}
        for window in self.config.bb_windows:
            for std in self.config.bb_stds[:2]:
                if window < len(close):
                    if VECTORBT_AVAILABLE:
                        try:
                            # VectorBT-optimized Bollinger Bands calculation
                            sma = rolling_mean(close, window=window)
                            std_dev = rolling_std(close, window=window)
                            upper_band = sma + (std_dev * std)
                            lower_band = sma - (std_dev * std)
                            bb_width = upper_band - lower_band
                            bb_position = (close - lower_band) / (bb_width + 1e-08)

                            bb_cache[(window, std)] = {
                                'sma': sma,
                                'upper': upper_band,
                                'lower': lower_band,
                                'width': bb_width,
                                'position': bb_position
                            }
                        except Exception as e:
                            self.logger.warning(f"VectorBT Bollinger Bands calculation failed: {e}, using pandas fallback")
                            bb_cache[(window, std)] = self._calculate_bollinger_position_vectorized(close, window, std)
                    else:
                        bb_cache[(window, std)] = self._calculate_bollinger_position_vectorized(close, window, std)

        # VECTORIZED: Compute all window combinations simultaneously
        for (window1, std1), bb_1 in bb_cache.items():
            for (window2, std2), bb_2 in bb_cache.items():
                if window1 != window2 and bb_1 is not None and bb_2 is not None:
                    # VECTORIZED: Bollinger Band position difference
                    if isinstance(bb_1, dict) and isinstance(bb_2, dict):
                        bb_diff = bb_1['position'] - bb_2['position']
                        bb_width_diff = bb_1['width'] - bb_2['width']
                        bb_upper_diff = bb_1['upper'] - bb_2['upper']
                        bb_lower_diff = bb_1['lower'] - bb_2['lower']

                        if self._is_valid_feature(bb_diff):
                            features[f'bb_position_diff_{window1}_{std1}_{window2}_{std2}'] = bb_diff
                        if self._is_valid_feature(bb_width_diff):
                            features[f'bb_width_diff_{window1}_{std1}_{window2}_{std2}'] = bb_width_diff
                        if self._is_valid_feature(bb_upper_diff):
                            features[f'bb_upper_diff_{window1}_{std1}_{window2}_{std2}'] = bb_upper_diff
                        if self._is_valid_feature(bb_lower_diff):
                            features[f'bb_lower_diff_{window1}_{std1}_{window2}_{std2}'] = bb_lower_diff
                    else:
                        # Fallback for simple position calculation
                        bb_diff = bb_1 - bb_2
                        if self._is_valid_feature(bb_diff):
                            features[f'bb_position_diff_{window1}_{std1}_{window2}_{std2}'] = bb_diff

        return features

    def generate_advanced_interaction_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None = None) -> dict[str, pd.Series]:
        """
        Generate advanced interaction features using VectorBT optimization with memory management.

        This method creates sophisticated interaction features that capture complex
        relationships between different timeframes and indicators while managing memory usage.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary of advanced interaction features
        """
        if not self._validate_input_data(price_data):
            return {}

        self.logger.info("🚀 Generating advanced interaction features with VectorBT optimization and memory management")

        # Memory optimization: Process data in chunks if large
        data_size = len(price_data)
        chunk_size = self.config.min_data_points * 10  # Process in chunks

        if data_size > chunk_size:
            return self._generate_advanced_interaction_features_chunked(price_data, volume_data, chunk_size)

        features = {}
        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}

        # Advanced momentum interactions with memory cleanup
        momentum_interactions = self._generate_advanced_momentum_interactions(price_components)
        features.update(momentum_interactions)
        del momentum_interactions  # Memory cleanup

        # Advanced volatility interactions with memory cleanup
        volatility_interactions = self._generate_advanced_volatility_interactions(price_components)
        features.update(volatility_interactions)
        del volatility_interactions  # Memory cleanup

        # Advanced volume interactions with memory cleanup
        if volume_data is not None:
            volume_interactions = self._generate_advanced_volume_interactions(price_components, volume_data)
            features.update(volume_interactions)
            del volume_interactions  # Memory cleanup

        # Advanced technical indicator interactions with memory cleanup
        tech_interactions = self._generate_advanced_technical_interactions(price_components)
        features.update(tech_interactions)
        del tech_interactions  # Memory cleanup

        # Advanced cross-timeframe interactions with memory cleanup
        cross_tf_interactions = self._generate_advanced_cross_timeframe_interactions(price_components)
        features.update(cross_tf_interactions)
        del cross_tf_interactions  # Memory cleanup

        # Data-driven interactions (if generator available)
        if self.interaction_generator:
            try:
                data_driven_interactions = self.generate_data_driven_interactions(price_data, volume_data)
                features.update(data_driven_interactions)
                self.logger.info(f"✅ Added {len(data_driven_interactions)} data-driven interactions")
            except Exception as e:
                self.logger.warning(f"⚠️ Data-driven interactions failed: {e}")

        valid_features = self._validate_features(features)
        self.logger.info(f"✅ Generated {len(valid_features)} advanced interaction features with memory optimization")

        return valid_features

    def _generate_advanced_interaction_features_chunked(self, price_data: pd.DataFrame,
                                                       volume_data: pd.DataFrame | None,
                                                       chunk_size: int) -> dict[str, pd.Series]:
        """Generate advanced interaction features in chunks for memory efficiency."""
        self.logger.info(f"🔄 Processing data in chunks of {chunk_size} for memory efficiency")

        all_features = {}
        total_chunks = (len(price_data) + chunk_size - 1) // chunk_size

        for i in range(0, len(price_data), chunk_size):
            chunk_end = min(i + chunk_size, len(price_data))
            chunk_data = price_data.iloc[i:chunk_end]
            chunk_volume = volume_data.iloc[i:chunk_end] if volume_data is not None else None

            self.logger.debug(f"Processing chunk {i//chunk_size + 1}/{total_chunks}")

            # Generate features for this chunk
            chunk_features = self._generate_advanced_interaction_features_single_chunk(chunk_data, chunk_volume)
            all_features.update(chunk_features)

            # Memory cleanup
            del chunk_data
            if chunk_volume is not None:
                del chunk_volume
            del chunk_features

        self.logger.info(f"✅ Completed chunked processing: {len(all_features)} features generated")
        return all_features

    def _generate_advanced_interaction_features_single_chunk(self, price_data: pd.DataFrame,
                                                           volume_data: pd.DataFrame | None) -> dict[str, pd.Series]:
        """Generate advanced interaction features for a single chunk."""
        features = {}
        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}

        # Process each type of interaction with memory management
        try:
            # Advanced momentum interactions
            momentum_interactions = self._generate_advanced_momentum_interactions(price_components)
            features.update(momentum_interactions)
        except Exception as e:
            self.logger.warning(f"Momentum interactions failed for chunk: {e}")

        try:
            # Advanced volatility interactions
            volatility_interactions = self._generate_advanced_volatility_interactions(price_components)
            features.update(volatility_interactions)
        except Exception as e:
            self.logger.warning(f"Volatility interactions failed for chunk: {e}")

        try:
            # Advanced volume interactions
            if volume_data is not None:
                volume_interactions = self._generate_advanced_volume_interactions(price_components, volume_data)
                features.update(volume_interactions)
        except Exception as e:
            self.logger.warning(f"Volume interactions failed for chunk: {e}")

        try:
            # Advanced technical indicator interactions
            tech_interactions = self._generate_advanced_technical_interactions(price_components)
            features.update(tech_interactions)
        except Exception as e:
            self.logger.warning(f"Technical interactions failed for chunk: {e}")

        try:
            # Advanced cross-timeframe interactions
            cross_tf_interactions = self._generate_advanced_cross_timeframe_interactions(price_components)
            features.update(cross_tf_interactions)
        except Exception as e:
            self.logger.warning(f"Cross-timeframe interactions failed for chunk: {e}")

        # Data-driven interactions (if generator available)
        if self.interaction_generator:
            try:
                # Create temporary DataFrame for data-driven interactions
                temp_data = pd.DataFrame({'close': price_components['close']})
                if volume_data is not None and 'volume' in volume_data.columns:
                    temp_data['volume'] = volume_data['volume']

                data_driven_interactions = self.generate_data_driven_interactions(temp_data, volume_data)
                features.update(data_driven_interactions)
            except Exception as e:
                self.logger.warning(f"Data-driven interactions failed for chunk: {e}")

        return features

    def get_data_driven_timeframes(self, data: pd.DataFrame, target_timeframe: str = "15m") -> List[int]:
        """
        Get data-driven timeframes based on data characteristics.

        Args:
            data: Input data for analysis
            target_timeframe: Target timeframe (e.g., "15m", "5m", "1h")

        Returns:
            List of optimal timeframes
        """
        # Try enhanced period selector first (with economic evaluation)
        if self.enhanced_period_selector:
            try:
                result = self.enhanced_period_selector.select_optimal_periods(data, target_timeframe)

                if result.optimal_periods:
                    self.logger.info(f"✅ Enhanced data-driven timeframes selected: {result.optimal_periods}")
                    self.logger.info(f"💰 Economic evaluation: {result.successful_evaluations} successful evaluations")
                    if result.economic_evaluation_result:
                        self.logger.info(f"📊 Best economic score: {result.best_score:.3f}")
                    return result.optimal_periods
                else:
                    self.logger.warning("⚠️ No optimal periods found with enhanced selector, trying basic selector")

            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced period selection failed: {e}, trying basic selector")

        # Fallback to basic period selector
        if self.period_selector:
            try:
                result = self.period_selector.select_optimal_periods(data, target_timeframe)

                if result.optimal_periods:
                    self.logger.info(f"✅ Basic data-driven timeframes selected: {result.optimal_periods}")
                    return result.optimal_periods
                else:
                    self.logger.warning("⚠️ No optimal periods found, using fallback")
                    return [15, 30, 60, 120]

            except Exception as e:
                self.logger.warning(f"⚠️ Basic period selection failed: {e}, using fallback")
                return [15, 30, 60, 120]

        # Final fallback to default timeframes
        self.logger.warning("⚠️ No period selector available, using default timeframes")
        return [15, 30, 60, 120]

    def generate_data_driven_interactions(self,
                                        price_data: pd.DataFrame,
                                        volume_data: pd.DataFrame | None = None,
                                        targets: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Generate data-driven interaction features using comprehensive exploration.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)
            targets: Target variable (optional)

        Returns:
            Dictionary of data-driven interaction features
        """
        if not self.interaction_generator:
            self.logger.warning("⚠️ Data-driven interaction generator not available")
            return {}

        if not self._validate_input_data(price_data):
            return {}

        self.logger.info("🚀 Generating data-driven interaction features")

        # Prepare feature data
        features_data = self._prepare_feature_data(price_data, volume_data)

        if features_data.empty:
            self.logger.warning("⚠️ No features available for interaction generation")
            return {}

        # Generate interactions
        interactions = self.interaction_generator.generate_interactions(features_data, targets)

        # Convert to dictionary format
        interaction_features = {}
        for interaction in interactions:
            interaction_features[interaction.feature_name] = interaction.feature_series

        self.logger.info(f"✅ Generated {len(interaction_features)} data-driven interaction features")

        return interaction_features

    def _prepare_feature_data(self,
                            price_data: pd.DataFrame,
                            volume_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Prepare feature data for interaction generation."""
        features = {}

        # Basic price features
        if 'close' in price_data.columns:
            close = price_data['close']
            features['close'] = close
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))

            # Price momentum
            features['momentum_5'] = close.pct_change(5)
            features['momentum_10'] = close.pct_change(10)
            features['momentum_20'] = close.pct_change(20)

            # Price volatility
            features['volatility_5'] = close.pct_change().rolling(5).std()
            features['volatility_10'] = close.pct_change().rolling(10).std()
            features['volatility_20'] = close.pct_change().rolling(20).std()

        # High-Low features
        if 'high' in price_data.columns and 'low' in price_data.columns:
            high = price_data['high']
            low = price_data['low']
            features['hl_range'] = high - low
            features['hl_ratio'] = high / (low + 1e-08)
            features['hl_position'] = (close - low) / (high - low + 1e-08)

        # Volume features
        if volume_data is not None and 'volume' in volume_data.columns:
            volume = volume_data['volume']
            features['volume'] = volume
            features['volume_ma_5'] = volume.rolling(5).mean()
            features['volume_ma_20'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / (volume.rolling(20).mean() + 1e-08)

        # Technical indicators
        if 'close' in price_data.columns:
            close = price_data['close']

            # RSI
            features['rsi_14'] = self._calculate_rsi_vectorized(close, 14)
            features['rsi_21'] = self._calculate_rsi_vectorized(close, 21)

            # MACD
            macd_line = self._calculate_macd_vectorized(close, 12, 26)
            if macd_line is not None:
                features['macd'] = macd_line
                features['macd_signal'] = macd_line.ewm(span=9, adjust=False).mean()
                features['macd_histogram'] = macd_line - features['macd_signal']

            # Bollinger Bands
            bb_window = 20
            bb_std = 2.0
            if VECTORBT_AVAILABLE:
                sma = rolling_mean(close, window=bb_window)
                std_dev = rolling_std(close, window=bb_window)
            else:
                sma = close.rolling(window=bb_window).mean()
                std_dev = close.rolling(window=bb_window).std()

            upper_band = sma + (std_dev * bb_std)
            lower_band = sma - (std_dev * bb_std)
            bb_width = upper_band - lower_band
            bb_position = (close - lower_band) / (bb_width + 1e-08)

            features['bb_position'] = bb_position
            features['bb_width'] = bb_width
            features['bb_squeeze'] = (bb_width < bb_width.rolling(20).mean() * 0.8).astype(float)

        # Create DataFrame and clean data
        features_df = pd.DataFrame(features, index=price_data.index)
        features_df = features_df.dropna()

        return features_df

    def generate_comprehensive_interaction_features(self,
                                                  price_data: pd.DataFrame,
                                                  volume_data: pd.DataFrame | None = None,
                                                  targets: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Generate comprehensive interaction features including all types.

        This method combines:
        - Cross-timeframe features (data-driven timeframes)
        - Advanced interaction features (RSI-MACD, Bollinger Bands, etc.)
        - Data-driven interactions (comprehensive exploration)

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)
            targets: Target variable (optional)

        Returns:
            Dictionary of comprehensive interaction features
        """
        self.logger.info("🚀 Generating comprehensive interaction features")

        all_features = {}

        # 1. Cross-timeframe features (data-driven timeframes)
        try:
            cross_tf_features = self.generate_cross_timeframe_features(price_data, volume_data)
            all_features.update(cross_tf_features)
            self.logger.info(f"✅ Added {len(cross_tf_features)} cross-timeframe features")
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-timeframe features failed: {e}")

        # 2. Advanced interaction features
        try:
            advanced_features = self.generate_advanced_interaction_features(price_data, volume_data)
            all_features.update(advanced_features)
            self.logger.info(f"✅ Added {len(advanced_features)} advanced interaction features")
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced interaction features failed: {e}")

        # 3. Data-driven interactions (if generator available)
        if self.interaction_generator:
            try:
                data_driven_features = self.generate_data_driven_interactions(price_data, volume_data, targets)
                all_features.update(data_driven_features)
                self.logger.info(f"✅ Added {len(data_driven_features)} data-driven interaction features")
            except Exception as e:
                self.logger.warning(f"⚠️ Data-driven interactions failed: {e}")

        # 4. Validate and filter features
        valid_features = self._validate_features(all_features)

        self.logger.info(f"✅ Generated {len(valid_features)} comprehensive interaction features")

        return valid_features

    def _generate_advanced_momentum_interactions(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate advanced momentum interaction features using VectorBT."""
        features = {}
        close = price_components['close']

        if not VECTORBT_AVAILABLE:
            return features

        try:
            # Multi-timeframe momentum convergence/divergence (data-driven timeframes)
            # Create a temporary DataFrame for period analysis
            temp_data = pd.DataFrame({'close': close})
            timeframes = self.get_data_driven_timeframes(temp_data, "15m")
            momentum_series = {}

            for tf in timeframes:
                if tf < len(close):
                    momentum_series[tf] = close.pct_change(tf)

            # Momentum convergence score
            if len(momentum_series) >= 3:
                momentum_df = pd.DataFrame(momentum_series)
                momentum_corr = momentum_df.corr()

                # Calculate convergence as average correlation
                convergence_score = momentum_corr.mean().mean()
                if self._is_valid_feature(convergence_score):
                    features['momentum_convergence_score'] = pd.Series(
                        [convergence_score] * len(close),
                        index=close.index,
                        name='momentum_convergence_score'
                    )

                # Momentum divergence detection
                momentum_std = momentum_df.std(axis=1)
                if self._is_valid_feature(momentum_std):
                    features['momentum_divergence_std'] = momentum_std.rename('momentum_divergence_std')

            # Momentum acceleration across timeframes
            if len(momentum_series) >= 2:
                short_momentum = momentum_series.get(5)
                long_momentum = momentum_series.get(20)

                if short_momentum is not None and long_momentum is not None:
                    # Acceleration as second derivative
                    momentum_acceleration = short_momentum.diff() - long_momentum.diff()
                    if self._is_valid_feature(momentum_acceleration):
                        features['momentum_acceleration'] = momentum_acceleration.rename('momentum_acceleration')

        except Exception as e:
            self.logger.warning(f"Advanced momentum interactions failed: {e}")

        return features

    def _generate_advanced_volatility_interactions(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate advanced volatility interaction features using VectorBT."""
        features = {}
        close = price_components['close']

        if not VECTORBT_AVAILABLE:
            return features

        try:
            # Multi-timeframe volatility analysis (data-driven timeframes)
            # Create a temporary DataFrame for period analysis
            temp_data = pd.DataFrame({'close': close})
            timeframes = self.get_data_driven_timeframes(temp_data, "15m")
            volatility_series = {}

            for tf in timeframes:
                if tf < len(close):
                    returns = close.pct_change()
                    if VECTORBT_AVAILABLE:
                        volatility_series[tf] = rolling_std(returns, window=tf)
                    else:
                        volatility_series[tf] = returns.rolling(window=tf).std()

            # Volatility regime detection
            if len(volatility_series) >= 2:
                vol_df = pd.DataFrame(volatility_series)

                # Volatility clustering
                vol_clustering = vol_df.rolling(window=10).corr().mean().mean()
                if self._is_valid_feature(vol_clustering):
                    features['volatility_clustering'] = pd.Series(
                        [vol_clustering] * len(close),
                        index=close.index,
                        name='volatility_clustering'
                    )

                # Volatility mean reversion
                short_vol = volatility_series.get(5)
                long_vol = volatility_series.get(20)

                if short_vol is not None and long_vol is not None:
                    vol_mean_reversion = (short_vol - long_vol) / (long_vol + 1e-08)
                    if self._is_valid_feature(vol_mean_reversion):
                        features['volatility_mean_reversion'] = vol_mean_reversion.rename('volatility_mean_reversion')

            # Volatility of volatility
            if len(volatility_series) >= 1:
                main_vol = list(volatility_series.values())[0]
                if VECTORBT_AVAILABLE:
                    vol_of_vol = rolling_std(main_vol, window=20)
                else:
                    vol_of_vol = main_vol.rolling(window=20).std()

                if self._is_valid_feature(vol_of_vol):
                    features['volatility_of_volatility'] = vol_of_vol.rename('volatility_of_volatility')

        except Exception as e:
            self.logger.warning(f"Advanced volatility interactions failed: {e}")

        return features

    def _generate_advanced_volume_interactions(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate advanced volume interaction features using VectorBT."""
        features = {}
        close = price_components['close']
        volume = volume_data['volume'] if 'volume' in volume_data.columns else volume_data.iloc[:, 0]

        if not VECTORBT_AVAILABLE:
            return features

        try:
            # Price-Volume interaction analysis
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Price-Volume correlation
            if VECTORBT_AVAILABLE:
                price_volume_corr = rolling_corr(price_change, volume_change, window=20)
            else:
                price_volume_corr = price_change.rolling(window=20).corr(volume_change)

            if self._is_valid_feature(price_volume_corr):
                features['price_volume_correlation'] = price_volume_corr.rename('price_volume_correlation')

            # Volume-weighted price momentum
            vwap = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
            price_vwap_ratio = close / (vwap + 1e-08)

            if self._is_valid_feature(price_vwap_ratio):
                features['price_vwap_ratio'] = price_vwap_ratio.rename('price_vwap_ratio')

            # Volume momentum divergence
            price_momentum = close.pct_change(5)
            volume_momentum = volume.pct_change(5)
            volume_divergence = price_momentum - volume_momentum

            if self._is_valid_feature(volume_divergence):
                features['volume_momentum_divergence'] = volume_divergence.rename('volume_momentum_divergence')

        except Exception as e:
            self.logger.warning(f"Advanced volume interactions failed: {e}")

        return features

    def _generate_advanced_technical_interactions(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate advanced technical indicator interaction features using VectorBT."""
        features = {}
        close = price_components['close']

        if not VECTORBT_AVAILABLE:
            return features

        try:
            # RSI-MACD interactions (multiple types)
            rsi_14 = self._calculate_rsi_vectorized(close, 14)
            rsi_21 = self._calculate_rsi_vectorized(close, 21)
            macd_line = self._calculate_macd_vectorized(close, 12, 26)
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - macd_signal

            if rsi_14 is not None and macd_line is not None:
                # RSI-MACD divergence (difference)
                rsi_macd_divergence = rsi_14 - macd_line
                if self._is_valid_feature(rsi_macd_divergence):
                    features['rsi_macd_divergence'] = rsi_macd_divergence.rename('rsi_macd_divergence')

                # RSI-MACD momentum (product)
                rsi_macd_momentum = rsi_14 * macd_line
                if self._is_valid_feature(rsi_macd_momentum):
                    features['rsi_macd_momentum'] = rsi_macd_momentum.rename('rsi_macd_momentum')

                # RSI-MACD ratio
                rsi_macd_ratio = rsi_14 / (macd_line + 1e-08)
                if self._is_valid_feature(rsi_macd_ratio):
                    features['rsi_macd_ratio'] = rsi_macd_ratio.rename('rsi_macd_ratio')

                # RSI-MACD correlation (rolling)
                if VECTORBT_AVAILABLE:
                    rsi_macd_corr = rolling_corr(rsi_14, macd_line, window=20)
                else:
                    rsi_macd_corr = rsi_14.rolling(window=20).corr(macd_line)
                if self._is_valid_feature(rsi_macd_corr):
                    features['rsi_macd_correlation'] = rsi_macd_corr.rename('rsi_macd_correlation')

                # RSI-MACD signal interaction
                rsi_macd_signal_interaction = rsi_14 * macd_signal
                if self._is_valid_feature(rsi_macd_signal_interaction):
                    features['rsi_macd_signal_interaction'] = rsi_macd_signal_interaction.rename('rsi_macd_signal_interaction')

                # RSI-MACD histogram interaction
                rsi_macd_histogram_interaction = rsi_14 * macd_histogram
                if self._is_valid_feature(rsi_macd_histogram_interaction):
                    features['rsi_macd_histogram_interaction'] = rsi_macd_histogram_interaction.rename('rsi_macd_histogram_interaction')

                # RSI-MACD normalized divergence
                rsi_macd_normalized = (rsi_14 - 50) * (macd_line - macd_line.rolling(50).mean())
                if self._is_valid_feature(rsi_macd_normalized):
                    features['rsi_macd_normalized'] = rsi_macd_normalized.rename('rsi_macd_normalized')

            # Multi-period RSI-MACD interactions
            if rsi_21 is not None and macd_line is not None:
                # RSI21-MACD interaction
                rsi21_macd_interaction = rsi_21 * macd_line
                if self._is_valid_feature(rsi21_macd_interaction):
                    features['rsi21_macd_interaction'] = rsi21_macd_interaction.rename('rsi21_macd_interaction')

                # RSI14-RSI21-MACD interaction
                rsi_diff_macd = (rsi_14 - rsi_21) * macd_line
                if self._is_valid_feature(rsi_diff_macd):
                    features['rsi_diff_macd_interaction'] = rsi_diff_macd.rename('rsi_diff_macd_interaction')

            # Bollinger Bands interactions (multiple types)
            bb_windows = [15, 20, 30]  # Multiple BB periods
            bb_stds = [1.5, 2.0, 2.5]  # Multiple standard deviations

            for bb_window in bb_windows:
                for bb_std in bb_stds:
                    if VECTORBT_AVAILABLE:
                        sma = rolling_mean(close, window=bb_window)
                        std_dev = rolling_std(close, window=bb_window)
                    else:
                        sma = close.rolling(window=bb_window).mean()
                        std_dev = close.rolling(window=bb_window).std()

                    upper_band = sma + (std_dev * bb_std)
                    lower_band = sma - (std_dev * bb_std)
                    bb_width = upper_band - lower_band
                    bb_position = (close - lower_band) / (bb_width + 1e-08)

                    # BB squeeze detection (low volatility)
                    bb_squeeze = bb_width < bb_width.rolling(window=20).mean() * 0.8
                    if self._is_valid_feature(bb_squeeze):
                        features[f'bb_squeeze_{bb_window}_{bb_std}'] = bb_squeeze.astype(float).rename(f'bb_squeeze_{bb_window}_{bb_std}')

                    # BB position (where price sits in bands)
                    if self._is_valid_feature(bb_position):
                        features[f'bb_position_{bb_window}_{bb_std}'] = bb_position.rename(f'bb_position_{bb_window}_{bb_std}')

                    # BB width (volatility measure)
                    if self._is_valid_feature(bb_width):
                        features[f'bb_width_{bb_window}_{bb_std}'] = bb_width.rename(f'bb_width_{bb_window}_{bb_std}')

                    # BB distance from middle band
                    bb_distance = close - sma
                    if self._is_valid_feature(bb_distance):
                        features[f'bb_distance_{bb_window}_{bb_std}'] = bb_distance.rename(f'bb_distance_{bb_window}_{bb_std}')

                    # BB normalized distance
                    bb_normalized = bb_distance / (std_dev + 1e-08)
                    if self._is_valid_feature(bb_normalized):
                        features[f'bb_normalized_{bb_window}_{bb_std}'] = bb_normalized.rename(f'bb_normalized_{bb_window}_{bb_std}')

                    # BB breakout detection
                    bb_breakout_upper = close > upper_band
                    bb_breakout_lower = close < lower_band
                    if self._is_valid_feature(bb_breakout_upper):
                        features[f'bb_breakout_upper_{bb_window}_{bb_std}'] = bb_breakout_upper.astype(float).rename(f'bb_breakout_upper_{bb_window}_{bb_std}')
                    if self._is_valid_feature(bb_breakout_lower):
                        features[f'bb_breakout_lower_{bb_window}_{bb_std}'] = bb_breakout_lower.astype(float).rename(f'bb_breakout_lower_{bb_window}_{bb_std}')

            # Cross-BB interactions
            if len(bb_windows) >= 2:
                # BB width ratio between different periods
                bb_width_15 = features.get('bb_width_15_2.0')
                bb_width_30 = features.get('bb_width_30_2.0')
                if bb_width_15 is not None and bb_width_30 is not None:
                    bb_width_ratio = bb_width_15 / (bb_width_30 + 1e-08)
                    if self._is_valid_feature(bb_width_ratio):
                        features['bb_width_ratio_15_30'] = bb_width_ratio.rename('bb_width_ratio_15_30')

                # BB position difference between periods
                bb_pos_15 = features.get('bb_position_15_2.0')
                bb_pos_30 = features.get('bb_position_30_2.0')
                if bb_pos_15 is not None and bb_pos_30 is not None:
                    bb_pos_diff = bb_pos_15 - bb_pos_30
                    if self._is_valid_feature(bb_pos_diff):
                        features['bb_position_diff_15_30'] = bb_pos_diff.rename('bb_position_diff_15_30')

            # BB-MACD interactions
            if macd_line is not None:
                bb_pos_20 = features.get('bb_position_20_2.0')
                if bb_pos_20 is not None:
                    # BB position * MACD
                    bb_macd_interaction = bb_pos_20 * macd_line
                    if self._is_valid_feature(bb_macd_interaction):
                        features['bb_position_macd_interaction'] = bb_macd_interaction.rename('bb_position_macd_interaction')

                    # BB squeeze * MACD
                    bb_squeeze_20 = features.get('bb_squeeze_20_2.0')
                    if bb_squeeze_20 is not None:
                        bb_squeeze_macd = bb_squeeze_20 * macd_line
                        if self._is_valid_feature(bb_squeeze_macd):
                            features['bb_squeeze_macd_interaction'] = bb_squeeze_macd.rename('bb_squeeze_macd_interaction')

            # BB-RSI interactions
            if rsi_14 is not None:
                bb_pos_20 = features.get('bb_position_20_2.0')
                if bb_pos_20 is not None:
                    # BB position * RSI
                    bb_rsi_interaction = bb_pos_20 * rsi_14
                    if self._is_valid_feature(bb_rsi_interaction):
                        features['bb_position_rsi_interaction'] = bb_rsi_interaction.rename('bb_position_rsi_interaction')

                    # BB squeeze * RSI
                    bb_squeeze_20 = features.get('bb_squeeze_20_2.0')
                    if bb_squeeze_20 is not None:
                        bb_squeeze_rsi = bb_squeeze_20 * rsi_14
                        if self._is_valid_feature(bb_squeeze_rsi):
                            features['bb_squeeze_rsi_interaction'] = bb_squeeze_rsi.rename('bb_squeeze_rsi_interaction')

        except Exception as e:
            self.logger.warning(f"Advanced technical interactions failed: {e}")

        return features

    def _generate_advanced_cross_timeframe_interactions(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate advanced cross-timeframe interaction features using VectorBT."""
        features = {}
        close = price_components['close']

        if not VECTORBT_AVAILABLE:
            return features

        try:
            # Multi-timeframe trend alignment (data-driven timeframes)
            # Create a temporary DataFrame for period analysis
            temp_data = pd.DataFrame({'close': close})
            timeframes = self.get_data_driven_timeframes(temp_data, "15m")
            trend_indicators = {}

            for tf in timeframes:
                if tf < len(close):
                    if VECTORBT_AVAILABLE:
                        sma = rolling_mean(close, window=tf)
                    else:
                        sma = close.rolling(window=tf).mean()

                    # Trend direction (1 for uptrend, -1 for downtrend, 0 for sideways)
                    trend_direction = np.where(close > sma, 1, np.where(close < sma, -1, 0))
                    trend_indicators[tf] = pd.Series(trend_direction, index=close.index)

            # Trend alignment score
            if len(trend_indicators) >= 3:
                trend_df = pd.DataFrame(trend_indicators)
                trend_alignment = trend_df.mean(axis=1)

                if self._is_valid_feature(trend_alignment):
                    features['trend_alignment_score'] = trend_alignment.rename('trend_alignment_score')

                # Trend consistency
                trend_consistency = trend_df.std(axis=1)
                if self._is_valid_feature(trend_consistency):
                    features['trend_consistency'] = trend_consistency.rename('trend_consistency')

            # Cross-timeframe momentum divergence
            if len(trend_indicators) >= 2:
                short_trend = trend_indicators.get(5)
                long_trend = trend_indicators.get(20)

                if short_trend is not None and long_trend is not None:
                    trend_divergence = short_trend - long_trend
                    if self._is_valid_feature(trend_divergence):
                        features['cross_timeframe_trend_divergence'] = trend_divergence.rename('cross_timeframe_trend_divergence')

        except Exception as e:
            self.logger.warning(f"Advanced cross-timeframe interactions failed: {e}")

        return features

    def _calculate_bollinger_position_vectorized(self, prices: pd.Series, window: int, num_std: float) -> pd.Series | None:
        """VECTORIZED: Calculate position relative to Bollinger Bands"""
        if len(prices) < window:
            return None

        # VECTORIZED: Bollinger Bands calculation
        sma = self._vectorbt_rolling_operation(prices, "mean", window)
        std = self._vectorbt_rolling_operation(prices, "std", window)
        upper_band = sma + std * num_std
        lower_band = sma - std * num_std

        # VECTORIZED: Position calculation
        return (prices - lower_band) / (upper_band - lower_band + 1e-08)

    def _generate_volume_features_vectorized(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame) -> dict[str, pd.Series]:
        """VECTORIZED: Generate volume-based cross-timeframe features"""
        features = {}
        if 'volume' not in volume_data.columns:
            return features

        volume = volume_data['volume'].astype(float)
        if volume.var() <= self.config.variance_threshold:
            return features

        timeframes = self.config.volume_timeframes[:3]

        # VECTORIZED: Pre-compute all volume calculations
        vol_cache = {}
        vol_pct_cache = {}
        for tf in timeframes:
            vol_cache[tf] = volume.rolling(tf, min_periods=tf//2).mean()
            vol_pct_cache[tf] = volume.pct_change(tf)

        # VECTORIZED: Compute all timeframe combinations simultaneously
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(volume) and tf2 < len(volume):
                    volume_features = self._calculate_volume_pair_vectorized(vol_cache, vol_pct_cache, tf1, tf2)
                    features.update(volume_features)

        return features

    def _calculate_volume_pair_vectorized(self, vol_cache: dict[int, pd.Series],
                                        vol_pct_cache: dict[int, pd.Series], tf1: int, tf2: int) -> dict[str, pd.Series]:
        """VECTORIZED: Calculate volume features for a timeframe pair"""
        features = {}

        vol_1 = vol_cache[tf1]
        vol_2 = vol_cache[tf2]

        # VECTORIZED: Volume ratio
        vol_ratio = vol_1 / (vol_2 + 1e-08)
        if self._is_valid_feature(vol_ratio):
            features[f'volume_ratio_{tf1}m_{tf2}m'] = vol_ratio

        # VECTORIZED: Volume difference
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f'volume_diff_{tf1}m_{tf2}m'] = vol_diff

        # VECTORIZED: Volume momentum difference
        vol_momentum_1 = vol_pct_cache[tf1]
        vol_momentum_2 = vol_pct_cache[tf2]
        vol_momentum = vol_momentum_1 - vol_momentum_2
        if self._is_valid_feature(vol_momentum):
            features[f'volume_momentum_{tf1}m_{tf2}m'] = vol_momentum

        return features

    async def _generate_features_with_pipeline(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None = None) -> dict[str, pd.Series]:
        """Generate features using the comprehensive cross timeframe analysis pipeline."""
        try:
            # Create a temporary data directory structure for the pipeline
            import tempfile
            import os
            from pathlib import Path

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

except ImportError:

    cp = None

            with tempfile.TemporaryDirectory() as temp_dir:
                # Save data to temporary parquet file
                temp_file = Path(temp_dir) / "temp_data.parquet"

                # Combine price and volume data
                combined_data = price_data.copy()
                if volume_data is not None:
                    combined_data['volume'] = volume_data['volume'] if 'volume' in volume_data.columns else volume_data.iloc[:, 0]
                else:
                    # Create mock volume data if not provided
                    combined_data['volume'] = 1000.0

                # Add timestamp if not present
                if 'timestamp' not in combined_data.columns:
                    combined_data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(combined_data), freq='1T')

                # Save to parquet
                combined_data.to_parquet(temp_file)

                # Run the cross timeframe analysis pipeline
                result = await self.cross_timeframe_pipeline.analyze_cross_timeframes(
                    data_dir=temp_dir,
                    symbol="TEMP",
                    exchange="temp",
                    timeframes=['1m', '5m', '15m', '30m']
                )

                # Convert the cross timeframe features to the expected format
                features = {}
                if hasattr(result, 'cross_timeframe_features') and not result.cross_timeframe_features.empty:
                    for col in result.cross_timeframe_features.columns:
                        features[col] = result.cross_timeframe_features[col]

                # Add interaction metrics as features
                if hasattr(result, 'interaction_metrics'):
                    for key, value in result.interaction_metrics.items():
                        if isinstance(value, (int, float)):
                            features[f'interaction_{key}'] = pd.Series([value] * len(price_data), index=price_data.index)

                # Add timeframe correlations as features
                if hasattr(result, 'timeframe_correlations'):
                    for metric, corr_matrix in result.timeframe_correlations.items():
                        if isinstance(corr_matrix, pd.DataFrame):
                            # Extract average correlation as a feature
                            avg_corr = corr_matrix.values.mean()
                            features[f'timeframe_corr_{metric}'] = pd.Series([avg_corr] * len(price_data), index=price_data.index)

                self.logger.info(f'✅ Generated {len(features)} features using comprehensive pipeline')
                return features

        except Exception as e:
            self.logger.error(f"❌ Pipeline-based feature generation failed: {e}")
            raise

    @log_all_calls

    def _validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """Validate input data meets requirements"""
        if price_data.empty or len(price_data) < self.config.min_data_points:
            self.logger.warning(f'⚠️ Insufficient data: {len(price_data)} rows, need at least {self.config.min_data_points}')
            return False
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(price_data.columns):
            self.logger.warning(f'⚠️ Missing required columns: {required_cols - set(price_data.columns)}')
            return False
        return True
    @log_all_calls

    def _extract_price_components(self, price_data: pd.DataFrame) -> dict[str, pd.Series]:
        """Extract and validate price components"""
        try:
            components = {'close': price_data['close'].astype(float), 'high': price_data['high'].astype(float), 'low': price_data['low'].astype(float), 'open': price_data['open'].astype(float)}
            if components['close'].isna().all() or components['close'].std() == 0:
                self.logger.warning('⚠️ Invalid close data')
                return {}
            return components
        except Exception as e:
            self.logger.exception(f'❌ Error extracting price components: {e}')
            return {}
    @log_all_calls

    def _generate_features_parallel(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame | None) -> dict[str, pd.Series]:
        """Generate features using parallel processing"""
        features = {}
        with ThreadPoolExecutor(max_workers = self.config.max_workers) as executor:
            futures = []
            futures.append(executor.submit(self._generate_momentum_features, price_components))
            futures.append(executor.submit(self._generate_volatility_features, price_components))
            futures.append(executor.submit(self._generate_range_features, price_components))
            futures.append(executor.submit(self._generate_technical_indicator_features, price_components))
            if volume_data is not None:
                futures.append(executor.submit(self._generate_volume_features, price_components, volume_data))
            for future in as_completed(futures):
                try:
                    result = future.result()
                    features.update(result)
                except Exception as e:
                    self.logger.exception(f'❌ Feature generation task failed: {e}')
        return features
    @log_all_calls

    def _generate_features_sequential(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame | None) -> dict[str, pd.Series]:
        """Generate features sequentially"""
        features = {}
        features.update(self._generate_momentum_features(price_components))
        features.update(self._generate_volatility_features(price_components))
        features.update(self._generate_range_features(price_components))
        features.update(self._generate_technical_indicator_features(price_components))
        if volume_data is not None:
            features.update(self._generate_volume_features(price_components, volume_data))
        return features
    @log_all_calls

    def _generate_momentum_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate momentum-based cross-timeframe features"""
        features = {}
        close = price_components['close']
        high = price_components['high']
        low = price_components['low']
        timeframes = self.config.momentum_timeframes[:4]
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    momentum_diff = close.pct_change(tf1) - close.pct_change(tf2)
                    if self._is_valid_feature(momentum_diff):
                        features[f'momentum_{tf1}m_{tf2}m'] = momentum_diff
                    momentum_ratio = close.pct_change(tf1) / (close.pct_change(tf2) + 1e-08)
                    if self._is_valid_feature(momentum_ratio):
                        features[f'momentum_ratio_{tf1}m_{tf2}m'] = momentum_ratio
                    if len(close) >= max(tf1, tf2) * 2:
                        hl_features = self._calculate_hl_momentum(high, low, close, tf1, tf2)
                        features.update(hl_features)
        return features
    @log_all_calls

    def _calculate_hl_momentum(self, high: pd.Series, low: pd.Series, close: pd.Series, tf1: int, tf2: int) -> dict[str, pd.Series]:
        """Calculate high-low momentum features"""
        features = {}
        hl_momentum_1 = (high.rolling(tf1, min_periods = tf1 // 2).max() - low.rolling(tf1, min_periods = tf1 // 2).min()) / (close.rolling(tf1, min_periods = tf1 // 2).mean() + 1e-08)
        hl_momentum_2 = (high.rolling(tf2, min_periods = tf2 // 2).max() - low.rolling(tf2, min_periods = tf2 // 2).min()) / (close.rolling(tf2, min_periods = tf2 // 2).mean() + 1e-08)
        hl_diff = hl_momentum_1 - hl_momentum_2
        if self._is_valid_feature(hl_diff):
            features[f'hl_momentum_{tf1}m_{tf2}m'] = hl_diff
        return features
    @log_all_calls

    def _generate_volatility_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate volatility-based cross-timeframe features"""
        features = {}
        close = price_components['close']
        returns = close.pct_change().fillna(method='ffill').fillna(method='bfill').fillna(0)
        timeframes = self.config.volatility_timeframes[:3]
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    vol_features = self._calculate_volatility_pair(returns, tf1, tf2)
                    features.update(vol_features)
        return features
    @log_all_calls

    def _calculate_volatility_pair(self, returns: pd.Series, tf1: int, tf2: int) -> dict[str, pd.Series]:
        """Calculate volatility features for a timeframe pair"""
        features = {}
        vol_1 = returns.rolling(tf1, min_periods = tf1 // 2).std()
        vol_2 = returns.rolling(tf2, min_periods = tf2 // 2).std()
        vol_ratio = vol_1 / (vol_2 + 1e-08)
        if self._is_valid_feature(vol_ratio):
            features[f'volatility_ratio_{tf1}m_{tf2}m'] = vol_ratio
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f'volatility_diff_{tf1}m_{tf2}m'] = vol_diff
        if len(returns) >= 20:
            vol_std = (vol_1 - vol_2).rolling(20, min_periods = 10).std()
            if self._is_valid_feature(vol_std):
                features[f'volatility_std_{tf1}m_{tf2}m'] = vol_std
        return features
    @log_all_calls

    def _generate_range_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate price range cross-timeframe features"""
        features = {}
        high = price_components['high']
        low = price_components['low']
        close = price_components['close']
        timeframes = self.config.momentum_timeframes[:3]
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    range_features = self._calculate_range_pair(high, low, close, tf1, tf2)
                    features.update(range_features)
        return features
    @log_all_calls

    def _calculate_range_pair(self, high: pd.Series, low: pd.Series, close: pd.Series, tf1: int, tf2: int) -> dict[str, pd.Series]:
        """Calculate range features for a timeframe pair"""
        features = {}
        range_1 = (high.rolling(tf1, min_periods = tf1 // 2).max() - low.rolling(tf1, min_periods = tf1 // 2).min()) / (close.rolling(tf1, min_periods = tf1 // 2).mean() + 1e-08)
        range_2 = (high.rolling(tf2, min_periods = tf2 // 2).max() - low.rolling(tf2, min_periods = tf2 // 2).min()) / (close.rolling(tf2, min_periods = tf2 // 2).mean() + 1e-08)
        range_ratio = range_1 / (range_2 + 1e-08)
        if self._is_valid_feature(range_ratio):
            features[f'price_range_ratio_{tf1}m_{tf2}m'] = range_ratio
        range_diff = range_1 - range_2
        if self._is_valid_feature(range_diff):
            features[f'price_range_diff_{tf1}m_{tf2}m'] = range_diff
        return features
    @log_all_calls

    def _generate_technical_indicator_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate technical indicator cross-timeframe features"""
        features = {}
        features.update(self._generate_rsi_features(price_components))
        features.update(self._generate_macd_features(price_components))
        features.update(self._generate_bb_features(price_components))
        return features
    @log_all_calls

    def _generate_rsi_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate RSI cross-timeframe features"""
        features = {}
        close = price_components['close']
        for i, period1 in enumerate(self.config.rsi_periods[:-1]):
            for period2 in self.config.rsi_periods[i + 1:]:
                if period1 < len(close) and period2 < len(close):
                    rsi_1 = self._calculate_rsi(close, period1)
                    rsi_2 = self._calculate_rsi(close, period2)
                    rsi_diff = rsi_1 - rsi_2
                    if self._is_valid_feature(rsi_diff):
                        features[f'rsi_diff_{period1}_{period2}'] = rsi_diff
                    rsi_ratio = rsi_1 / (rsi_2 + 1e-08)
                    if self._is_valid_feature(rsi_ratio):
                        features[f'rsi_ratio_{period1}_{period2}'] = rsi_ratio
        return features
    @log_all_calls

    def _generate_macd_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate MACD cross-timeframe features"""
        features = {}
        close = price_components['close']
        for fast in self.config.macd_fast_periods[:3]:
            for slow in self.config.macd_slow_periods[:3]:
                if fast < slow < len(close):
                    macd_1 = self._calculate_macd(close, fast, slow)
                    macd_2 = self._calculate_macd(close, fast * 2, slow * 2)
                    macd_diff = macd_1 - macd_2
                    if self._is_valid_feature(macd_diff):
                        features[f'macd_diff_{fast}_{slow}'] = macd_diff
                    macd_ratio = macd_1 / (macd_2 + 1e-08)
                    if self._is_valid_feature(macd_ratio):
                        features[f'macd_ratio_{fast}_{slow}'] = macd_ratio
        return features
    @log_all_calls

    def _generate_bb_features(self, price_components: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Generate Bollinger Bands cross-timeframe features"""
        features = {}
        close = price_components['close']
        for window in self.config.bb_windows:
            for std in self.config.bb_stds[:2]:
                if window < len(close):
                    bb_1 = self._calculate_bollinger_position(close, window, std)
                    bb_2 = self._calculate_bollinger_position(close, window * 2, std)
                    if bb_1 is not None and bb_2 is not None:
                        bb_diff = bb_1 - bb_2
                        if self._is_valid_feature(bb_diff):
                            features[f'bb_position_diff_{window}_{std}'] = bb_diff
        return features
    @log_all_calls

    def _generate_volume_features(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate volume-based cross-timeframe features"""
        features = {}
        if 'volume' not in volume_data.columns:
            return features
        volume = volume_data['volume'].astype(float)
        if volume.var() <= self.config.variance_threshold:
            return features
        timeframes = self.config.volume_timeframes[:3]
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(volume) and tf2 < len(volume):
                    volume_features = self._calculate_volume_pair(volume, tf1, tf2)
                    features.update(volume_features)
        return features
    @log_all_calls

    def _calculate_volume_pair(self, volume: pd.Series, tf1: int, tf2: int) -> dict[str, pd.Series]:
        """Calculate volume features for a timeframe pair"""
        features = {}
        vol_1 = volume.rolling(tf1, min_periods = tf1 // 2).mean()
        vol_2 = volume.rolling(tf2, min_periods = tf2 // 2).mean()
        vol_ratio = vol_1 / (vol_2 + 1e-08)
        if self._is_valid_feature(vol_ratio):
            features[f'volume_ratio_{tf1}m_{tf2}m'] = vol_ratio
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f'volume_diff_{tf1}m_{tf2}m'] = vol_diff
        vol_momentum = volume.pct_change(tf1) - volume.pct_change(tf2)
        if self._is_valid_feature(vol_momentum):
            features[f'volume_momentum_{tf1}m_{tf2}m'] = vol_momentum
        return features
    @log_all_calls

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window = period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
        rs = gain / (loss + 1e-08)
        return 100 - 100 / (1 + rs)
    @log_all_calls

    def _calculate_macd(self, prices: pd.Series, fast_period: int, slow_period: int) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span = fast_period, adjust = False).mean()
        exp2 = prices.ewm(span = slow_period, adjust = False).mean()
        return exp1 - exp2
    @log_all_calls

    def _calculate_bollinger_position(self, prices: pd.Series, window: int, num_std: float) -> pd.Series | None:
        """Calculate position relative to Bollinger Bands"""
        try:
            sma = prices.rolling(window = window).mean()
            std = prices.rolling(window = window).std()
            upper_band = sma + std * num_std
            lower_band = sma - std * num_std
            return (prices - lower_band) / (upper_band - lower_band + 1e-08)
        except Exception:
            return None
    @log_all_calls

    def _is_valid_feature(self, feature: pd.Series) -> bool:
        """Check if a feature is valid"""
        if feature is None or feature.empty:
            return False
        if feature.var() <= self.config.variance_threshold:
            return False
        return not feature.isna().all()
    @log_all_calls

    def _validate_features(self, features: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Validate and filter features"""
        valid_features = {}
        for name, feature in features.items():
            if self._is_valid_feature(feature):
                valid_features[name] = feature
            else:
                self.logger.debug(f'⚠️ Skipping invalid feature: {name}')
        return valid_features

class InteractionFeatureGenerator:
    """Refactored interaction feature generator with reduced complexity"""
    @log_important_calls

    def __init__(self, config: InteractionConfig | None = None, logger: logging.Logger | None = None) -> None:
        """Initialize the generator.

        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or InteractionConfig()
        self.logger = logger or logging.getLogger(__name__)

    def generate_interaction_features(self, features: pd.DataFrame, feature_categories: dict[str, list[str]] | None = None) -> pd.DataFrame:
        """Generate interaction features with reduced complexity.

        Args:
            features: DataFrame containing base features
            feature_categories: Optional categorization of features

        Returns:
            DataFrame containing interaction features
        """
        if features.empty:
            self.logger.warning('⚠️ Empty features provided')
            return pd.DataFrame()
        selected_features = self._select_top_features(features)
        if len(selected_features) < 2:
            self.logger.warning('⚠️ Not enough features for interactions')
            return pd.DataFrame()
        if self.config.parallel_processing:
            interaction_features = self._generate_interactions_parallel(features[selected_features])
        else:
            interaction_features = self._generate_interactions_sequential(features[selected_features])
        final_features = self._remove_correlated_features(interaction_features)
        self.logger.info(f'✅ Generated {len(final_features.columns)} interaction features')
        return final_features
    @log_all_calls

    def _select_top_features(self, features: pd.DataFrame) -> list[str]:
        """Select top features based on variance"""
        variances = features.var()
        valid_features = variances[variances > self.config.variance_threshold]
        return valid_features.nlargest(self.config.top_k_features).index.tolist()
    @log_all_calls

    def _generate_interactions_parallel(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interactions using parallel processing"""
        interaction_dfs = []
        with ThreadPoolExecutor(max_workers = self.config.max_workers) as executor:
            futures = []
            if self.config.include_ratios:
                futures.append(executor.submit(self._generate_ratio_features, features))
            if self.config.include_differences:
                futures.append(executor.submit(self._generate_difference_features, features))
            if self.config.include_products:
                futures.append(executor.submit(self._generate_product_features, features))
            if self.config.polynomial_degree > 1:
                futures.append(executor.submit(self._generate_polynomial_features, features))
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty:
                        interaction_dfs.append(result)
                except Exception as e:
                    self.logger.exception(f'❌ Interaction generation failed: {e}')
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis = 1)
        return pd.DataFrame()
    @log_all_calls

    def _generate_interactions_sequential(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interactions sequentially"""
        interaction_dfs = []
        if self.config.include_ratios:
            interaction_dfs.append(self._generate_ratio_features(features))
        if self.config.include_differences:
            interaction_dfs.append(self._generate_difference_features(features))
        if self.config.include_products:
            interaction_dfs.append(self._generate_product_features(features))
        if self.config.polynomial_degree > 1:
            interaction_dfs.append(self._generate_polynomial_features(features))
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis = 1)
        return pd.DataFrame()
    @log_all_calls

    def _generate_ratio_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio interaction features"""
        ratio_features = pd.DataFrame(index = features.index)
        feature_cols = features.columns.tolist()
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                if self._same_category(col1, col2):
                    continue
                ratio = features[col1] / (features[col2] + 1e-08)
                if self._is_valid_interaction(ratio):
                    ratio_name = f'{col1}_ratio_{col2}'
                    ratio_features[ratio_name] = ratio
        return ratio_features
    @log_all_calls

    def _generate_difference_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate difference interaction features"""
        diff_features = pd.DataFrame(index = features.index)
        feature_cols = features.columns.tolist()
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                if self._same_category(col1, col2):
                    continue
                diff = features[col1] - features[col2]
                if self._is_valid_interaction(diff):
                    diff_name = f'{col1}_diff_{col2}'
                    diff_features[diff_name] = diff
        return diff_features
    @log_all_calls

    def _generate_product_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate product interaction features"""
        product_features = pd.DataFrame(index = features.index)
        feature_cols = features.columns.tolist()
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                if self._same_category(col1, col2):
                    continue
                product = features[col1] * features[col2]
                if self._is_valid_interaction(product):
                    product_name = f'{col1}_x_{col2}'
                    product_features[product_name] = product
        return product_features
    @log_all_calls

    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial interaction features"""
        poly_features = pd.DataFrame(index = features.index)
        for col in features.columns:
            for degree in range(2, self.config.polynomial_degree + 1):
                poly = features[col] ** degree
                if self._is_valid_interaction(poly):
                    poly_name = f'{col}_pow{degree}'
                    poly_features[poly_name] = poly
        return poly_features
    @log_all_calls

    def _same_category(self, col1: str, col2: str) -> bool:
        """Check if two columns belong to the same category"""
        cat1 = col1.split('_')[0]
        cat2 = col2.split('_')[0]
        same_categories = {('ma', 'ema', 'sma'), ('rsi', 'rsi'), ('macd', 'macd'), ('bb', 'bollinger'), ('volume', 'vol')}
        for category_group in same_categories:
            if cat1 in category_group and cat2 in category_group:
                return True
        return False
    @log_all_calls

    def _is_valid_interaction(self, feature: pd.Series) -> bool:
        """Check if an interaction feature is valid"""
        if feature.empty:
            return False
        if feature.var() <= self.config.variance_threshold:
            return False
        return not (feature.isna().all() or np.isinf(feature).any())
    @log_all_calls

    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if features.empty:
            return features
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config.correlation_threshold)]
        result = features.drop(columns = to_drop)
        self.logger.info(f'Removed {len(to_drop)} highly correlated features')
        return result
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
