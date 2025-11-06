"""
Feature Engineering for Rolling HMM Clustering

This module implements comprehensive feature engineering with EWMA-style rolling windows,
including returns, volatility, trend, and volume features. Optimized for Mac M1 with
VectorBT and hardware acceleration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
import logging
from enum import Enum
from numba import jit
import numpy.typing as npt

# Type aliases
ArrayLike = Union[npt.NDArray[np.float64], pd.Series, pd.DataFrame]
Numeric = Union[int, float, np.number]
WindowType = Union[int, str, pd.api.indexers.BaseIndexer]
NormalizeMethod = Literal['zscore', 'robust']

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_debug, tprint_error
from src.feature_generation.core.feature_generator import FeatureResult
from src.feature_generation.utils.consolidated_rolling_optimizer import (
    ConsolidatedRollingOptimizer,
    BatchRollingConfig,
    RollingOperationConfig,
    RollingOperationType
)
from src.feature_generation.utils.statistical_calculations_optimizer import (
    StatisticalCalculationsOptimizer,
    BatchStatisticalConfig,
    StatisticalOperationConfig,
    StatisticalOperationType
)
from src.features_common.normalization import (
    RollingZScoreGenerator,
    RollingRobustGenerator
)
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager,
    WorkloadType,
    OptimizationLevel
)

logger = logging.getLogger(__name__)


class EWMAConfig:
    """Configuration for EWMA feature generation."""
    
    def __init__(
        self,
        short_window: int,
        long_window: int,
        name: str
    ) -> None:
        self.short_window = short_window  # e.g., 8, 12
        self.long_window = long_window    # e.g., 16, 20, 24
        self.name = name                 # e.g., "8+16", "12+24"
        self.__post_init__()
        
    def __post_init__(self) -> None:
        if self.short_window >= self.long_window:
            tprint_error(
                f"⚠️  Invalid EWMAConfig: short_window={self.short_window} must be < long_window={self.long_window}"
            )
            raise ValueError(f"short_window ({self.short_window}) must be < long_window ({self.long_window})")


class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""
    
    def __init__(
        self,
        ewma_configs: List[EWMAConfig],
        use_log_returns: bool = True,
        use_volatility_features: bool = True,
        use_trend_features: bool = True,
        use_volume_features: bool = True,
        pca_components: int = 4,  # 3-5 for 80-90% variance
        normalize_method: NormalizeMethod = 'zscore',
        rolling_normalize_window: int = 100,
        enable_vectorbt_optimization: bool = True,
        enable_hardware_optimization: bool = True,
        enable_numba_jit: bool = True
    ) -> None:
        self.ewma_configs = ewma_configs
        self.use_log_returns = use_log_returns
        self.use_volatility_features = use_volatility_features
        self.use_trend_features = use_trend_features
        self.use_volume_features = use_volume_features
        self.pca_components = pca_components
        self.normalize_method = normalize_method
        self.rolling_normalize_window = rolling_normalize_window
        self.enable_vectorbt_optimization = enable_vectorbt_optimization
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_numba_jit = enable_numba_jit


class RollingHMMFeatureEngineer:
    """
    Feature engineering for Rolling HMM Clustering.

    Generates comprehensive features using EWMA rolling windows with multiple periods,
    including returns, volatility, trend, and volume features. Optimized with VectorBT
    and hardware acceleration for Mac M1.
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Feature cache for all EWMA windows (pre-computed once, reused in HPO)
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._normalized_feature_cache: Dict[str, pd.DataFrame] = {}
        self._pca_cache: Dict[Tuple[str, int], Tuple[pd.DataFrame, Any, float]] = {}
        
        # Initialize optimizers with proper type hints
        self.rolling_optimizer: Optional[ConsolidatedRollingOptimizer] = None
        self.stat_optimizer: Optional[StatisticalCalculationsOptimizer] = None
        self.hardware_manager = get_unified_hardware_manager() if config.enable_hardware_optimization else None

        # Initialize optimizers
        if config.enable_vectorbt_optimization:
            tprint_info("🚀 Initializing VectorBT optimizers")
            self.rolling_optimizer = ConsolidatedRollingOptimizer(
                BatchRollingConfig(
                    enable_gpu=True,
                    enable_parallel=True,
                    memory_optimization=True,
                    performance_threshold=100
                )
            )
            self.statistical_optimizer = StatisticalCalculationsOptimizer(
                BatchStatisticalConfig(
                    enable_gpu=True,
                    enable_parallel=True,
                    memory_optimization=True,
                    performance_threshold=1000
                )
            )
        else:
            self.rolling_optimizer = None
            self.statistical_optimizer = None

        # Initialize hardware optimization
        if config.enable_hardware_optimization:
            tprint_info("⚡ Enabling hardware optimization for M1")
            self.hardware_manager = get_unified_hardware_manager()
            self.hardware_manager.optimize_for_workload(
                WorkloadType.FEATURE_ENGINEERING,
                OptimizationLevel.BALANCED
            )
        else:
            self.hardware_manager = None

        # Initialize normalizers
        if config.normalize_method == 'zscore':
            self.normalizer = RollingZScoreGenerator(
                rolling_window=config.rolling_normalize_window
            )
        elif config.normalize_method == 'robust':
            self.normalizer = RollingRobustGenerator(
                rolling_window=config.rolling_normalize_window
            )
        else:
            raise ValueError(f"Unknown normalize_method: {config.normalize_method}")

        self.feature_names = []

    @staticmethod
    def _ensure_series(value: Any, index: pd.Index, name: str) -> pd.Series:
        """Convert batch results into a pandas Series aligned with the target index."""
        if value is None:
            series = pd.Series(np.nan, index=index, name=name)
        elif isinstance(value, pd.Series):
            series = value.copy()
        elif isinstance(value, pd.DataFrame):
            series = value.iloc[:, 0].copy()
        else:
            series = pd.Series(value, index=index, name=name)

        if not series.index.equals(index):
            series = series.reindex(index)

        series.name = name
        return series

    @staticmethod
    def _extract_batch_results(raw_results: Any, configs: List[RollingOperationConfig]) -> List[Any]:
        """Normalize batch operation outputs into a simple list."""
        result_list: List[Any]

        if isinstance(raw_results, dict):
            result_list = []
            values_iter = iter(raw_results.values())
            for idx, cfg in enumerate(configs):
                key = f"{cfg.operation.value}_{cfg.window}_{idx}"
                if key in raw_results:
                    result_list.append(raw_results[key])
                else:
                    try:
                        result_list.append(next(values_iter))
                    except StopIteration:
                        result_list.append(None)
        elif isinstance(raw_results, (list, tuple)):
            result_list = list(raw_results)
        else:
            result_list = [raw_results]

        if len(result_list) < len(configs):
            result_list.extend([None] * (len(configs) - len(result_list)))
        elif len(result_list) > len(configs):
            result_list = result_list[:len(configs)]

        return result_list

    def precompute_all_features(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Pre-compute features for ALL EWMA windows ONCE at the beginning.
        This cache is then reused throughout all HPO trials for efficiency.

        Args:
            market_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dictionary mapping EWMA config names to normalized feature DataFrames
        """
        tprint("🔄 Pre-computing features for ALL EWMA windows (will be cached for HPO)")
        tprint(f"   → Computing {len(self.config.ewma_configs)} EWMA configurations")

        all_features = {}

        for i, ewma_config in enumerate(self.config.ewma_configs, 1):
            tprint_info(f"   [{i}/{len(self.config.ewma_configs)}] Computing features for EWMA {ewma_config.name}")

            # Generate features for this EWMA config
            features = self._generate_features_internal(market_data, ewma_config)

            empty_columns = [col for col in features.columns if features[col].isna().all()]
            if empty_columns:
                display_cols = ", ".join(empty_columns[:10])
                suffix = " …" if len(empty_columns) > 10 else ""
                tprint_warning(
                    f"      ⚠️  EWMA {ewma_config.name}: Dropping {len(empty_columns)} all-NaN columns: {display_cols}{suffix}"
                )
                features = features.drop(columns=empty_columns)

            if features.empty:
                tprint_error(
                    f"      ❌ EWMA {ewma_config.name}: No features remaining after removing all-NaN columns; skipping"
                )
                continue

            # Normalize features
            features_normalized = self._normalize_features(features)

            total_rows = len(features_normalized)
            nan_mask = features_normalized.isna().any(axis=1)
            nan_rows = int(np.count_nonzero(nan_mask.to_numpy()))
            if nan_rows > 0:
                tprint_warning(
                    f"      ⚠️  EWMA {ewma_config.name}: Found {nan_rows} NaN rows out of {total_rows}; filling with column means"
                )
                features_normalized = features_normalized.fillna(features_normalized.mean())
                features_normalized = features_normalized.fillna(0)
            else:
                tprint_info(
                    f"      ✅ EWMA {ewma_config.name}: No NaNs detected across {total_rows} rows"
                )

            # Cache both raw and normalized features
            self._feature_cache[ewma_config.name] = features
            self._normalized_feature_cache[ewma_config.name] = features_normalized
            all_features[ewma_config.name] = features_normalized

            tprint_info(f"      ✓ Cached {len(features_normalized.columns)} features, {len(features_normalized)} samples")

        tprint(f"✅ Pre-computed and cached features for {len(all_features)} EWMA windows")
        return all_features

    def get_cached_features(self, ewma_config: EWMAConfig) -> Optional[pd.DataFrame]:
        """Get pre-computed features from cache."""
        if ewma_config is None:
            tprint_warning("⚠️  Requested cached features without providing an EWMA configuration")
            return None

        cached = self._normalized_feature_cache.get(ewma_config.name)

        if cached is None:
            tprint_debug(f"📦 Cache miss for EWMA {ewma_config.name}")
        else:
            tprint_debug(f"📦 Cache hit for EWMA {ewma_config.name} ({len(cached)} samples)")

        return cached

    def generate_features(
        self,
        market_data: pd.DataFrame,
        ewma_config: Optional[EWMAConfig] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Generate comprehensive features for HMM clustering.
        If features are already cached, return from cache instead of recomputing.

        Args:
            market_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            ewma_config: EWMA configuration (if None, uses first config from self.config)
            use_cache: If True and features are cached, return from cache

        Returns:
            DataFrame with engineered features
        """
        if ewma_config is None:
            ewma_config = self.config.ewma_configs[0]

        # Check cache first
        if use_cache and ewma_config.name in self._normalized_feature_cache:
            tprint_info(f"📦 Using cached features for EWMA {ewma_config.name}")
            return self._normalized_feature_cache[ewma_config.name]

        tprint(f"📊 Generating features for HMM clustering (EWMA: {ewma_config.name})")

        # Generate features internally
        features = self._generate_features_internal(market_data, ewma_config)

        empty_columns = [col for col in features.columns if features[col].isna().all()]
        if empty_columns:
            tprint_warning(
                f"  ⚠️  Dropping {len(empty_columns)} feature columns with only NaNs: {', '.join(empty_columns[:10])}" +
                (" …" if len(empty_columns) > 10 else "")
            )
            features = features.drop(columns=empty_columns)

        if features.empty:
            raise ValueError("No features remaining after removing all-NaN columns")

        # Store feature names before normalization
        self.feature_names = list(features.columns)

        # Normalize features
        tprint_info(f"  → Normalizing features ({self.config.normalize_method})")
        features_normalized = self._normalize_features(features)

        # Report NaN statistics before dropping
        total_rows = len(features_normalized)
        nan_mask = features_normalized.isna().any(axis=1)
        nan_rows = int(np.count_nonzero(nan_mask.to_numpy()))
        if nan_rows > 0:
            tprint_warning(
                f"  ⚠️  Found {nan_rows} rows with NaNs out of {total_rows}; filling with column means"
            )
            features_normalized = features_normalized.fillna(features_normalized.mean())
            features_normalized = features_normalized.fillna(0)
        else:
            tprint_info(f"  ✅ No NaNs detected across {total_rows} rows before drop")

        tprint(f"✅ Generated {len(features_normalized.columns)} features, {len(features_normalized)} samples")

        # Cache for future use
        self._normalized_feature_cache[ewma_config.name] = features_normalized
        self._feature_cache[ewma_config.name] = features

        return features_normalized

    def _generate_features_internal(
        self,
        market_data: pd.DataFrame,
        ewma_config: EWMAConfig
    ) -> pd.DataFrame:
        """
        Internal method to generate features (called by both generate_features and precompute_all_features).

        Args:
            market_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            ewma_config: EWMA configuration

        Returns:
            DataFrame with raw (unnormalized) features
        """
        features = {}

        # 1. Returns features
        tprint_info(f"  → Generating returns features (log={self.config.use_log_returns})")
        returns_features = self._generate_returns_features(market_data, ewma_config)
        features.update(returns_features)

        # 2. Volatility features
        if self.config.use_volatility_features:
            tprint_info(f"  → Generating volatility features (window={ewma_config.name})")
            volatility_features = self._generate_volatility_features(market_data, ewma_config)
            features.update(volatility_features)

        # 3. Trend features
        if self.config.use_trend_features:
            tprint_info(f"  → Generating trend features (window={ewma_config.name})")
            trend_features = self._generate_trend_features(market_data, ewma_config)
            features.update(trend_features)

        # 4. Volume features
        if self.config.use_volume_features:
            tprint_info(f"  → Generating volume features (window={ewma_config.name})")
            volume_features = self._generate_volume_features(market_data, ewma_config)
            features.update(volume_features)

        # Combine all features
        feature_df = pd.DataFrame(features, index=market_data.index)

        return feature_df

    def _generate_returns_features(
        self,
        market_data: pd.DataFrame,
        ewma_config: EWMAConfig
    ) -> Dict[str, pd.Series]:
        """Generate returns-based features."""
        tprint_debug(f"    Generating returns features for EWMA {ewma_config.name}")
        features = {}

        close = market_data['close']

        # Calculate returns (log or simple)
        if self.config.use_log_returns:
            returns = np.log(close / close.shift(1))
            features['log_returns'] = returns
        else:
            returns = close.pct_change()
            features['returns'] = returns

        # EWMA returns
        if self.rolling_optimizer:
            # Use VectorBT for batch operations
            ewma_short = returns.ewm(span=ewma_config.short_window, adjust=False).mean()
            ewma_long = returns.ewm(span=ewma_config.long_window, adjust=False).mean()
        else:
            ewma_short = returns.ewm(span=ewma_config.short_window, adjust=False).mean()
            ewma_long = returns.ewm(span=ewma_config.long_window, adjust=False).mean()

        features[f'ewma_returns_{ewma_config.short_window}'] = ewma_short
        features[f'ewma_returns_{ewma_config.long_window}'] = ewma_long
        features[f'ewma_returns_diff_{ewma_config.name}'] = ewma_short - ewma_long

        # Cumulative returns over windows
        features[f'cum_returns_{ewma_config.short_window}'] = returns.rolling(
            ewma_config.short_window
        ).sum()
        features[f'cum_returns_{ewma_config.long_window}'] = returns.rolling(
            ewma_config.long_window
        ).sum()

        return features

    def _generate_volatility_features(
        self,
        market_data: pd.DataFrame,
        ewma_config: EWMAConfig
    ) -> Dict[str, pd.Series]:
        """Generate volatility-based features."""
        tprint_debug(f"    Generating volatility features for EWMA {ewma_config.name}")
        features = {}

        close = market_data['close']
        high = market_data['high']
        low = market_data['low']

        # Calculate returns for volatility
        if self.config.use_log_returns:
            returns = np.log(close / close.shift(1))

        # Rolling standard deviation
        if self.rolling_optimizer:
            configs = [
                RollingOperationConfig(
                    operation=RollingOperationType.STD,
                    window=ewma_config.short_window,
                    min_periods=max(2, ewma_config.short_window // 2)
                ),
                RollingOperationConfig(
                    operation=RollingOperationType.STD,
                    window=ewma_config.long_window,
                    min_periods=max(2, ewma_config.long_window // 2)
                )
            ]
            raw_results = self.rolling_optimizer.batch_rolling_operations(returns, configs)
            result_list = self._extract_batch_results(raw_results, configs)
            vol_short = self._ensure_series(result_list[0], returns.index, f"rolling_std_{ewma_config.short_window}")
            vol_long = self._ensure_series(result_list[1], returns.index, f"rolling_std_{ewma_config.long_window}")
        else:
            vol_short = returns.rolling(
                ewma_config.short_window,
                min_periods=max(2, ewma_config.short_window // 2)
            ).std()
            vol_long = returns.rolling(
                ewma_config.long_window,
                min_periods=max(2, ewma_config.long_window // 2)
            ).std()

        features[f'volatility_{ewma_config.short_window}'] = vol_short
        features[f'volatility_{ewma_config.long_window}'] = vol_long
        features[f'volatility_ratio_{ewma_config.name}'] = vol_short / (vol_long + 1e-8)

        # EWMA volatility
        ewma_vol_short = returns.ewm(span=ewma_config.short_window, adjust=False).std()
        ewma_vol_long = returns.ewm(span=ewma_config.long_window, adjust=False).std()

        features[f'ewma_volatility_{ewma_config.short_window}'] = ewma_vol_short
        features[f'ewma_volatility_{ewma_config.long_window}'] = ewma_vol_long

        # Realized volatility (Parkinson estimator)
        hl_ratio = (high / low).apply(np.log)
        realized_vol = hl_ratio.rolling(ewma_config.short_window).std() * np.sqrt(252)
        features[f'realized_volatility_{ewma_config.short_window}'] = realized_vol

        # Log volatility (stabilizes scale)
        features[f'log_volatility_{ewma_config.short_window}'] = np.log(vol_short + 1e-8)
        features[f'log_volatility_{ewma_config.long_window}'] = np.log(vol_long + 1e-8)

        return features

    def _generate_trend_features(
        self,
        market_data: pd.DataFrame,
        ewma_config: EWMAConfig
    ) -> Dict[str, pd.Series]:
        """Generate trend-based features."""
        tprint_debug(f"    Generating trend features for EWMA {ewma_config.name}")
        features = {}

        close = market_data['close']

        # Calculate returns
        if self.config.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # Simple Moving Averages
        if self.rolling_optimizer:
            configs = [
                RollingOperationConfig(
                    operation=RollingOperationType.MEAN,
                    window=ewma_config.short_window,
                    min_periods=max(1, ewma_config.short_window // 2)
                ),
                RollingOperationConfig(
                    operation=RollingOperationType.MEAN,
                    window=ewma_config.long_window,
                    min_periods=max(1, ewma_config.long_window // 2)
                )
            ]
            raw_results = self.rolling_optimizer.batch_rolling_operations(close, configs)
            result_list = self._extract_batch_results(raw_results, configs)
            sma_short = self._ensure_series(result_list[0], close.index, f"sma_{ewma_config.short_window}")
            sma_long = self._ensure_series(result_list[1], close.index, f"sma_{ewma_config.long_window}")
        else:
            sma_short = close.rolling(
                ewma_config.short_window,
                min_periods=max(1, ewma_config.short_window // 2)
            ).mean()
            sma_long = close.rolling(
                ewma_config.long_window,
                min_periods=max(1, ewma_config.long_window // 2)
            ).mean()

        features[f'sma_{ewma_config.short_window}'] = sma_short
        features[f'sma_{ewma_config.long_window}'] = sma_long
        features[f'sma_diff_{ewma_config.name}'] = sma_short - sma_long
        features[f'price_to_sma_{ewma_config.short_window}'] = close / (sma_short + 1e-8)
        features[f'price_to_sma_{ewma_config.long_window}'] = close / (sma_long + 1e-8)

        # EWMA (Exponential Weighted Moving Average)
        ewma_short = close.ewm(span=ewma_config.short_window, adjust=False).mean()
        ewma_long = close.ewm(span=ewma_config.long_window, adjust=False).mean()

        features[f'ewma_{ewma_config.short_window}'] = ewma_short
        features[f'ewma_{ewma_config.long_window}'] = ewma_long
        features[f'ewma_diff_{ewma_config.name}'] = ewma_short - ewma_long
        features[f'price_to_ewma_{ewma_config.short_window}'] = close / (ewma_short + 1e-8)

        # Moving average slope
        sma_short_slope = sma_short.diff(ewma_config.short_window // 2)
        sma_long_slope = sma_long.diff(ewma_config.long_window // 2)

        features[f'sma_slope_{ewma_config.short_window}'] = sma_short_slope
        features[f'sma_slope_{ewma_config.long_window}'] = sma_long_slope

        # Rolling Sharpe ratio (annualized)
        rolling_mean = returns.rolling(ewma_config.short_window).mean()
        rolling_std = returns.rolling(ewma_config.short_window).std()
        rolling_sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)

        features[f'rolling_sharpe_{ewma_config.short_window}'] = rolling_sharpe

        # Rolling Z-score (mean-reversion indicator)
        rolling_zscore = (close - sma_short) / (rolling_std * close + 1e-8)
        features[f'rolling_zscore_{ewma_config.short_window}'] = rolling_zscore

        return features

    def _generate_volume_features(
        self,
        market_data: pd.DataFrame,
        ewma_config: EWMAConfig
    ) -> Dict[str, pd.Series]:
        """Generate volume-based features."""
        tprint_debug(f"    Generating volume features for EWMA {ewma_config.name}")
        features = {}

        volume = market_data['volume']
        close = market_data['close']

        # Log volume (stabilizes scale)
        log_volume = np.log(volume + 1)
        features['log_volume'] = log_volume

        # Rolling average volume
        if self.rolling_optimizer:
            configs = [
                RollingOperationConfig(
                    operation=RollingOperationType.MEAN,
                    window=ewma_config.short_window,
                    min_periods=max(1, ewma_config.short_window // 2)
                ),
                RollingOperationConfig(
                    operation=RollingOperationType.MEAN,
                    window=ewma_config.long_window,
                    min_periods=max(1, ewma_config.long_window // 2)
                )
            ]
            raw_results = self.rolling_optimizer.batch_rolling_operations(volume, configs)
            result_list = self._extract_batch_results(raw_results, configs)
            avg_vol_short = self._ensure_series(result_list[0], volume.index, f"avg_volume_{ewma_config.short_window}")
            avg_vol_long = self._ensure_series(result_list[1], volume.index, f"avg_volume_{ewma_config.long_window}")
        else:
            avg_vol_short = volume.rolling(
                ewma_config.short_window,
                min_periods=max(1, ewma_config.short_window // 2)
            ).mean()
            avg_vol_long = volume.rolling(
                ewma_config.long_window,
                min_periods=max(1, ewma_config.long_window // 2)
            ).mean()

        features[f'avg_volume_{ewma_config.short_window}'] = avg_vol_short
        features[f'avg_volume_{ewma_config.long_window}'] = avg_vol_long
        features[f'volume_ratio_{ewma_config.name}'] = volume / (avg_vol_short + 1e-8)

        # EWMA volume
        ewma_vol = volume.ewm(span=ewma_config.short_window, adjust=False).mean()
        features[f'ewma_volume_{ewma_config.short_window}'] = ewma_vol

        # Volume Z-score
        volume_mean = volume.rolling(ewma_config.long_window).mean()
        volume_std = volume.rolling(ewma_config.long_window).std()
        volume_zscore = (volume - volume_mean) / (volume_std + 1e-8)
        features[f'volume_zscore_{ewma_config.long_window}'] = volume_zscore

        # Volume changes
        volume_change = volume.pct_change()
        features['volume_change'] = volume_change

        # EMA of volume changes
        ewma_vol_change = volume_change.ewm(span=ewma_config.short_window, adjust=False).mean()
        features[f'ewma_volume_change_{ewma_config.short_window}'] = ewma_vol_change

        # Volume-weighted returns
        if self.config.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        vol_weighted_returns = returns * (volume / (avg_vol_short + 1e-8))
        features[f'volume_weighted_returns_{ewma_config.short_window}'] = vol_weighted_returns

        # On-Balance Volume (OBV) - cumulative volume weighted by price direction
        obv = self._calculate_obv(close, volume)
        features['obv'] = obv
        features[f'obv_ewma_{ewma_config.short_window}'] = obv.ewm(
            span=ewma_config.short_window, adjust=False
        ).mean()

        return features

    @staticmethod
    @jit(nopython=True)
    def _calculate_obv_numba(close_values: np.ndarray, volume_values: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume using Numba JIT."""
        n = len(close_values)
        obv = np.zeros(n)

        for i in range(1, n):
            if close_values[i] > close_values[i-1]:
                obv[i] = obv[i-1] + volume_values[i]
            elif close_values[i] < close_values[i-1]:
                obv[i] = obv[i-1] - volume_values[i]
            else:
                obv[i] = obv[i-1]

        return obv

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        if self.config.enable_numba_jit:
            tprint_debug("    Calculating OBV using Numba acceleration")
        else:
            tprint_debug("    Calculating OBV using pandas fallback")

        if self.config.enable_numba_jit:
            obv_values = self._calculate_obv_numba(close.values, volume.values)
            return pd.Series(obv_values, index=close.index)
        else:
            # Pandas fallback
            obv = pd.Series(0.0, index=close.index)
            obv.iloc[0] = 0

            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            return obv

    def _normalize_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using rolling z-score or robust scaling.

        Features with much larger variance will dominate the HMM fit unless scaled.
        """
        tprint_info(f"    Normalizing with {self.config.normalize_method}")

        normalized_features = {}

        for col in feature_df.columns:
            feature_data = feature_df[[col]]

            normalized_result = self.normalizer.generate(feature_data)

            normalized_series: Optional[pd.Series] = None

            if isinstance(normalized_result, FeatureResult):
                if normalized_result.success and normalized_result.data is not None:
                    data = normalized_result.data
                    if isinstance(data, pd.Series):
                        normalized_series = data
                    elif isinstance(data, pd.DataFrame):
                        if col in data.columns:
                            normalized_series = data[col]
                        elif data.shape[1] == 1:
                            normalized_series = data.iloc[:, 0]
                        else:
                            normalized_series = None
                else:
                    error_msg = normalized_result.error_message or "unknown error"
                    tprint_warning(
                        f"    ⚠️  Normalization for {col} failed ({error_msg}); using raw feature"
                    )
            elif isinstance(normalized_result, pd.Series):
                normalized_series = normalized_result
            elif isinstance(normalized_result, pd.DataFrame):
                candidate_names = [
                    col,
                    f"{col}_normalized",
                    f"{col}_zscore",
                    f"{col}_robust"
                ]
                for candidate in candidate_names:
                    if candidate in normalized_result.columns:
                        normalized_series = normalized_result[candidate]
                        break
                if normalized_series is None and normalized_result.shape[1] == 1:
                    normalized_series = normalized_result.iloc[:, 0]

            if isinstance(normalized_series, pd.DataFrame):
                if col in normalized_series.columns:
                    normalized_series = normalized_series[col]
                elif normalized_series.shape[1] == 1:
                    normalized_series = normalized_series.iloc[:, 0]
                else:
                    normalized_series = None

            if normalized_series is None or not isinstance(normalized_series, pd.Series):
                normalized_series = feature_df[col]
            else:
                normalized_series = normalized_series.reindex(feature_df.index)

            normalized_features[col] = normalized_series.rename(col)

        return pd.DataFrame(normalized_features, index=feature_df.index)

    def apply_pca(
        self,
        features: pd.DataFrame,
        n_components: Optional[int] = None,
        explained_variance_target: float = 0.85,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Any, float]:
        """
        Apply PCA for dimensionality reduction.
        Results are cached to avoid recomputation during HPO trials.

        Args:
            features: Feature DataFrame
            n_components: Number of components (if None, uses config.pca_components)
            explained_variance_target: Target explained variance (0.80-0.90)
            use_cache: If True, use cached PCA if available
            cache_key: Key for caching (if None, uses features index hash)

        Returns:
            Tuple of (transformed features, pca model, explained variance ratio)
        """
        from sklearn.decomposition import PCA

        if n_components is None:
            n_components = self.config.pca_components

        # Generate cache key
        if cache_key is None:
            cache_key = str(hash(tuple(features.index)))

        pca_cache_key = (cache_key, n_components)

        # Check cache
        if use_cache and pca_cache_key in self._pca_cache:
            tprint_info(f"  📦 Using cached PCA (n_components={n_components})")
            return self._pca_cache[pca_cache_key]

        tprint_info(f"  → Applying PCA (n_components={n_components})")

        # Ensure PCA input has no NaNs
        total_rows = len(features)
        nan_mask = features.isna().any(axis=1)
        nan_rows = int(np.count_nonzero(nan_mask.to_numpy()))
        if nan_rows > 0:
            tprint_warning(
                f"  ⚠️  PCA input contains {nan_rows} NaN rows out of {total_rows}; filling with column means"
            )
            features = features.fillna(features.mean())
            features = features.fillna(0)
        else:
            tprint_info(f"  ✅ PCA input has no NaNs across {total_rows} rows")

        if features.empty:
            raise ValueError("No data remaining after preprocessing for PCA")

        # Fit PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features.values)

        # Create DataFrame
        pca_columns = [f'pca_{i+1}' for i in range(n_components)]
        features_pca_df = pd.DataFrame(
            features_pca,
            index=features.index,
            columns=pca_columns
        )

        # Check explained variance
        explained_variance = pca.explained_variance_ratio_.sum()

        tprint_info(f"    PCA explained variance: {explained_variance:.2%}")

        if explained_variance < explained_variance_target:
            tprint_warning(
                f"    ⚠️  Explained variance ({explained_variance:.2%}) < "
                f"target ({explained_variance_target:.2%})"
            )

        # Cache result
        result = (features_pca_df, pca, explained_variance)
        self._pca_cache[pca_cache_key] = result

        return features_pca_df, pca, explained_variance


# Default EWMA configurations
DEFAULT_EWMA_CONFIGS = [
    EWMAConfig(short_window=8, long_window=16, name="8+16"),
    EWMAConfig(short_window=8, long_window=20, name="8+20"),
    EWMAConfig(short_window=8, long_window=24, name="8+24"),
    EWMAConfig(short_window=12, long_window=16, name="12+16"),
    EWMAConfig(short_window=12, long_window=20, name="12+20"),
    EWMAConfig(short_window=12, long_window=24, name="12+24"),
]
