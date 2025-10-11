from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

'\nRefactored cross-timeframe and interaction feature generation with reduced complexity.\nThis module breaks down the high-complexity feature generation methods into smaller,\nfocused functions with proper type annotations.\n'
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import typing

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
        """VECTORIZED: Generate MACD cross-timeframe features"""
        features = {}
        close = price_components['close']

        # VECTORIZED: Pre-compute all MACD calculations
        macd_cache = {}
        fast_periods = self.config.macd_fast_periods[:3]
        slow_periods = self.config.macd_slow_periods[:3]

        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow < len(close):
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
        """VECTORIZED: Generate Bollinger Bands cross-timeframe features"""
        features = {}
        close = price_components['close']

        # VECTORIZED: Pre-compute all Bollinger Band calculations
        bb_cache = {}
        for window in self.config.bb_windows:
            for std in self.config.bb_stds[:2]:
                if window < len(close):
                    bb_cache[(window, std)] = self._calculate_bollinger_position_vectorized(close, window, std)

        # VECTORIZED: Compute all window combinations simultaneously
        for (window1, std1), bb_1 in bb_cache.items():
            for (window2, std2), bb_2 in bb_cache.items():
                if window1 != window2 and bb_1 is not None and bb_2 is not None:
                    # VECTORIZED: Bollinger Band position difference
                    bb_diff = bb_1 - bb_2
                    if self._is_valid_feature(bb_diff):
                        features[f'bb_position_diff_{window1}_{std1}_{window2}_{std2}'] = bb_diff

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
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
