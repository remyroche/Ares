"""
Advanced Data Loading and Management for Unified Data-Driven Pipeline.

This module provides comprehensive data loading infrastructure similar to
FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Import utility modules
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation, validate_dataframe_columns,
    analyze_nan_values_detailed, format_nan_analysis_report,
    calculate_data_quality_metrics, create_data_quality_report,
    safe_convert_dtypes, safe_merge_dataframes, get_dataframe_info,
    create_summary_statistics, safe_drop_columns, safe_rename_columns
)
from src.utils.serialization_utils import UniversalSerializer
from src.utils.kline_parquet import KlinesParquetManager, StorageConfig

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_info
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)

import numpy as np
import pandas as pd

# Try to import feature cache service
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FEATURE_CACHE_AVAILABLE = False
    tprint_warning("⚠️ FeatureCacheService not available")

# ARES launcher integration not currently available
ARES_LAUNCHER_AVAILABLE = False

# Dummy class for ares launcher integration (placeholder for future implementation)
class AresLauncherFeatureLookbackOptimizer:
    """Placeholder class for Ares launcher feature lookback optimizer."""

    def __init__(self, *args, **kwargs):
        pass

    async def load_data_for_optimization(self, *args, **kwargs):
        """Placeholder method."""
        return None

class AdvancedDataLoader:
    """
    Advanced data loader for unified pipeline.

    Provides comprehensive data loading, caching, and management capabilities
    similar to FeatureLookbackOptimizationComponent.
    """

    def __init__(self, logger=None, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced data loader."""
        self.logger = logger or logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()
        self.config = config or {}

        # Initialize KlinesParquetManager for efficient klines data storage
        klines_config = self.config.get('klines_storage', {})
        storage_config = StorageConfig(
            base_dir=klines_config.get('base_dir', 'historical_data'),
            compression=klines_config.get('compression', 'zstd'),
            compression_level=klines_config.get('compression_level', 3),
            enable_metadata=klines_config.get('enable_metadata', True),
            enable_validation=klines_config.get('enable_validation', True),
            max_file_size_mb=klines_config.get('max_file_size_mb', 100)
        )
        self.klines_manager = KlinesParquetManager(storage_config)
        tprint_success("✅ KlinesParquetManager initialized")

        # Initialize feature cache if available
        if FEATURE_CACHE_AVAILABLE:
            self.feature_cache = FeatureCacheService(subdirectory="unified_pipeline")
            tprint_success("✅ Feature cache initialized")
        else:
            self.feature_cache = None
            tprint_warning("⚠️ Feature cache not available")

        # Initialize ares launcher integration if available
        if ARES_LAUNCHER_AVAILABLE:
            self.ares_integration = None  # Placeholder for future integration
            tprint_success("✅ Ares launcher integration initialized")
        else:
            self.ares_integration = None
            tprint_warning("⚠️ Ares launcher integration not available")

        # Cache metrics
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0,
            'load_times': [],
            'save_times': []
        }

        # Data loading configuration
        self.data_loading_config = {
            'default_timeframe': '15m',
            'default_exchange': 'binance',
            'default_symbol': 'ETHUSDT',
            'cache_ttl_hours': 24,
            'max_retries': 3,
            'retry_delay_seconds': 1
        }

        tprint_success("✅ AdvancedDataLoader initialized")

    async def load_market_data(self, data: Optional[pd.DataFrame] = None,
                             pipeline_state: Optional[Dict[str, Any]] = None,
                             force_refresh: bool = False) -> pd.DataFrame:
        """
        Load market data for the unified pipeline.

        Args:
            data: Optional pre-loaded data
            pipeline_state: Pipeline state with configuration
            force_refresh: Whether to force refresh cached data

        Returns:
            Loaded market data as DataFrame
        """
        tprint_debug("📥 Starting market data loading")

        # If data is already provided, validate and process it properly
        if data is not None and not data.empty:
            tprint_info(f"📊 Provided data detected: {data.shape[0]} rows, {data.shape[1]} columns")

            # Validate provided data
            if not self._validate_provided_data(data):
                tprint_warning("⚠️ Provided data validation failed, falling back to fresh data loading")
            else:
                # Apply data processing to provided data
                processed_data = await self._process_provided_data(data, pipeline_state)
                if processed_data is not None and not processed_data.empty:
                    tprint_success(f"✅ Using validated and processed provided data: {processed_data.shape}")
                    return processed_data
                else:
                    tprint_warning("⚠️ Data processing failed on provided data, falling back to fresh loading")

        # Extract configuration from pipeline state
        config = self._extract_data_config(pipeline_state)
        tprint_debug(f"📊 Data config: {config}")

        # Try to load from cache first
        if not force_refresh and self.feature_cache:
            cached_data = await self._load_from_cache(config)
            if cached_data is not None:
                tprint_success("✅ Loaded data from cache")
                return cached_data

        # Load data using ares integration or fallback
        if self.ares_integration:
            tprint_debug("🚀 Using ares launcher integration for data loading")
            try:
                market_data = await self._load_with_ares_integration(config)
                if market_data is not None and not market_data.empty:
                    # Cache the loaded data
                    if self.feature_cache:
                        await self._save_to_cache(market_data, config)
                    tprint_success(f"✅ Loaded data via ares integration: {market_data.shape}")
                    return market_data
            except Exception as e:
                tprint_warning(f"⚠️ Ares integration failed: {e}")

        # Fallback to synthetic data generation
        tprint_warning("⚠️ Using synthetic data generation as fallback")
        synthetic_data = self._generate_synthetic_data(config)
        tprint_success(f"✅ Generated synthetic data: {synthetic_data.shape}")
        return synthetic_data

    async def load_labeling_data(self, symbol: str, exchange: str, timeframe: str,
                               pipeline_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Load labeling data for supervised learning.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            pipeline_state: Pipeline state

        Returns:
            Labeling data dictionary or None
        """
        tprint_debug(f"📊 Loading labeling data for {symbol} on {exchange} ({timeframe})")

        # Try to load from pipeline state first
        if pipeline_state:
            labeling_data = pipeline_state.get('labeling_results')
            if labeling_data:
                tprint_success("✅ Found labeling data in pipeline state")
                return labeling_data

        # Try to load from cache
        if self.feature_cache:
            cache_key = f"labeling_{symbol}_{exchange}_{timeframe}"
            try:
                cached_data = self.feature_cache.get(cache_key)
                if cached_data:
                    tprint_success("✅ Loaded labeling data from cache")
                    return cached_data
            except Exception as e:
                tprint_warning(f"⚠️ Cache load failed: {e}")

        # Generate synthetic labeling data as fallback
        tprint_warning("⚠️ Generating synthetic labeling data as fallback")
        synthetic_labels = self._generate_synthetic_labels(symbol, exchange, timeframe)
        return synthetic_labels

    async def generate_features_for_optimization(self, market_data: pd.DataFrame,
                                               pipeline_state: Optional[Dict[str, Any]] = None,
                                               force_refresh: bool = False) -> List[str]:
        """
        Generate features for optimization using feature bank integration.

        Args:
            market_data: Market data DataFrame
            pipeline_state: Pipeline state
            force_refresh: Whether to force refresh feature generation

        Returns:
            List of generated feature column names
        """
        tprint_debug("🏦 Starting feature generation for optimization")

        if not self.feature_cache:
            tprint_warning("⚠️ Feature cache not available, using basic feature generation")
            return self._generate_basic_features(market_data)

        # Create cache key for features
        config = self._extract_data_config(pipeline_state)
        cache_key = self._create_feature_cache_key(market_data, config)

        # Try to load from cache first
        if not force_refresh:
            try:
                cached_features = self.feature_cache.get(cache_key)
                if cached_features:
                    tprint_success(f"✅ Loaded {len(cached_features)} features from cache")
                    return cached_features
            except Exception as e:
                tprint_warning(f"⚠️ Cache load failed: {e}")

        # Generate features using feature bank integration
        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.feature_bank_integration import FeatureBankIntegration

            feature_bank_config = {
                'enable_feature_bank': True,
                'enable_caching': True,
                'enable_multi_horizon': True,
                'enable_memory_optimization': True,
                'min_variance': 1e-8,
                'max_correlation_threshold': 0.95,
                'cache_force_refresh': force_refresh,
                'memory_efficient': True,
                'enable_parallel_processing': True,
                'max_workers': 4
            }

            feature_bank = FeatureBankIntegration(feature_bank_config)
            feature_result = feature_bank.generate_features_for_optimization(
                market_data, force_refresh=force_refresh
            )

            if feature_result.success:
                feature_columns = list(feature_result.feature_data.columns)
                tprint_success(f"✅ Generated {len(feature_columns)} features using feature bank")

                # Cache the features
                try:
                    self.feature_cache.set(cache_key, feature_columns)
                    tprint_debug("✅ Cached generated features")
                except Exception as e:
                    tprint_warning(f"⚠️ Feature caching failed: {e}")

                return feature_columns
            else:
                tprint_error(f"❌ Feature bank generation failed: {feature_result.error_message}")
                return self._generate_basic_features(market_data)

        except ImportError:
            tprint_warning("⚠️ FeatureBankIntegration not available, using basic features")
            return self._generate_basic_features(market_data)
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return self._generate_basic_features(market_data)

    def prepare_data_for_optimization(self, market_data: pd.DataFrame,
                                    labeling_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Prepare data for optimization by merging market data with labels using enhanced utilities.

        Args:
            market_data: Market data DataFrame
            labeling_data: Optional labeling data dictionary

        Returns:
            Prepared optimization data
        """
        tprint_debug("🧰 Preparing data for optimization with enhanced utilities")

        # Validate input data using utilities
        if not validate_dataframe_columns(market_data, ['close']):
            tprint_warning("⚠️ Market data missing required columns")

        optimization_data = market_data.copy()

        # Perform data quality analysis
        tprint_debug("📊 Analyzing data quality before optimization")
        nan_analysis = analyze_nan_values_detailed(optimization_data)
        quality_metrics = calculate_data_quality_metrics(optimization_data)

        # Log data quality report
        tprint_debug(format_nan_analysis_report(nan_analysis, "  "))

        # Add labeling data if available using enhanced merge operations
        if labeling_data and isinstance(labeling_data, dict):
            labels_df = labeling_data.get('labeled_data')
            if labels_df is not None and isinstance(labels_df, pd.DataFrame):
                # Use enhanced merge operation with better index handling
                try:
                    optimization_data, merge_stats = self._enhanced_label_merge(
                        optimization_data, labels_df
                    )

                    if merge_stats['success']:
                        tprint_success(f"✅ Merged {merge_stats['columns_added']} label columns "
                                     f"({merge_stats['rows_merged']} rows)")
                    else:
                        tprint_warning(f"⚠️ Label merging failed: {merge_stats['error']}")

                except Exception as e:
                    tprint_warning(f"⚠️ Label merging failed: {e}")
            else:
                tprint_warning("⚠️ No valid labeled data found in labeling_data")
        else:
            tprint_warning("⚠️ No labeling data provided")

        # Final data quality check
        final_quality = calculate_data_quality_metrics(optimization_data)
        tprint_success(f"✅ Prepared optimization data: {optimization_data.shape}")
        tprint_success(f"📊 Final data quality: {final_quality.get('missing_percentage', 0):.1f}% missing, {final_quality.get('duplicate_percentage', 0):.1f}% duplicates")

        return optimization_data

    def _extract_data_config(self, pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract data configuration from pipeline state."""
        config = self.data_loading_config.copy()

        if pipeline_state:
            config.update({
                'symbol': pipeline_state.get('symbol', config['default_symbol']),
                'exchange': pipeline_state.get('exchange', config['default_exchange']),
                'timeframe': pipeline_state.get('timeframe', config['default_timeframe']),
                'lookback_days': pipeline_state.get('lookback_days', 30),
                'execution_mode': pipeline_state.get('execution_mode', 'full')
            })

        return config

    async def _load_from_cache(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        if not self.feature_cache:
            return None

        cache_key = f"market_data_{config['symbol']}_{config['exchange']}_{config['timeframe']}"

        try:
            start_time = time.time()
            cached_data = self.feature_cache.get(cache_key)
            load_time = time.time() - start_time

            self.cache_metrics['load_times'].append(load_time)
            self.cache_metrics['hits'] += 1

            return cached_data
        except Exception as e:
            tprint_warning(f"⚠️ Cache load failed: {e}")
            self.cache_metrics['misses'] += 1
            return None

    async def _save_to_cache(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Save data to cache."""
        if not self.feature_cache:
            return

        cache_key = f"market_data_{config['symbol']}_{config['exchange']}_{config['timeframe']}"

        try:
            start_time = time.time()
            self.feature_cache.set(cache_key, data)
            save_time = time.time() - start_time

            self.cache_metrics['save_times'].append(save_time)
            self.cache_metrics['writes'] += 1

            tprint_debug(f"✅ Cached data with key: {cache_key}")
        except Exception as e:
            tprint_warning(f"⚠️ Cache save failed: {e}")

    async def _load_with_ares_integration(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load data using ares launcher integration."""
        if not self.ares_integration:
            return None

        try:
            # Create pipeline state for ares integration
            ares_pipeline_state = {
                'symbol': config['symbol'],
                'exchange': config['exchange'],
                'timeframe': config['timeframe'],
                'lookback_days': config['lookback_days'],
                'execution_mode': config['execution_mode']
            }

            # Load data
            market_data = await self.ares_integration.load_data_for_optimization(
                config['symbol'],
                config['timeframe'],
                ares_pipeline_state
            )

            return market_data
        except Exception as e:
            tprint_warning(f"⚠️ Ares integration data loading failed: {e}")
            return None

    def _generate_synthetic_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        tprint_debug("🔧 Generating synthetic market data")

        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['lookback_days'])
        date_range = pd.date_range(start=start_date, end=end_date, freq=config['timeframe'])

        # Generate synthetic OHLCV data with proper seed management
        # Note: This should be called with a seed manager in production
        np.random.seed(42)  # For reproducibility
        n_periods = len(date_range)

        # Generate price series with random walk
        base_price = 100.0
        returns = np.random.normal(0, 0.02, n_periods)  # 2% daily volatility
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)

        # Generate OHLCV
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_periods)
        }

        df = pd.DataFrame(data, index=date_range)

        # Ensure high >= max(open, close) and low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))

        return df

    def _generate_synthetic_labels(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate synthetic labeling data."""
        tprint_debug("🔧 Generating synthetic labeling data")

        # Create synthetic labels
        synthetic_labels = {
            'labeled_data': pd.DataFrame({
                'target_long': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
                'target_short': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
                'confidence_long': np.random.uniform(0.5, 1.0, 1000),
                'confidence_short': np.random.uniform(0.5, 1.0, 1000)
            }),
            'metadata': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'generated_at': datetime.now().isoformat(),
                'synthetic': True
            }
        }

        return synthetic_labels

    def _generate_basic_features(self, market_data: pd.DataFrame) -> List[str]:
        """Generate basic features as fallback."""
        tprint_debug("🔧 Generating basic features")

        feature_columns = []

        # Price-based features
        if 'close' in market_data.columns:
            market_data['sma_5'] = market_data['close'].rolling(5).mean()
            market_data['sma_20'] = market_data['close'].rolling(20).mean()
            market_data['rsi_14'] = self._calculate_rsi(market_data['close'], 14)
            market_data['bb_upper'] = market_data['close'].rolling(20).mean() + 2 * market_data['close'].rolling(20).std()
            market_data['bb_lower'] = market_data['close'].rolling(20).mean() - 2 * market_data['close'].rolling(20).std()

            feature_columns.extend(['sma_5', 'sma_20', 'rsi_14', 'bb_upper', 'bb_lower'])

        # Volume-based features
        if 'volume' in market_data.columns:
            market_data['volume_sma_10'] = market_data['volume'].rolling(10).mean()
            market_data['volume_ratio'] = market_data['volume'] / market_data['volume_sma_10']

            feature_columns.extend(['volume_sma_10', 'volume_ratio'])

        # Volatility features
        if 'close' in market_data.columns:
            market_data['volatility_10'] = market_data['close'].pct_change().rolling(10).std()
            market_data['volatility_20'] = market_data['close'].pct_change().rolling(20).std()

            feature_columns.extend(['volatility_10', 'volatility_20'])

        # Remove NaN values
        market_data[feature_columns] = market_data[feature_columns].fillna(method='bfill')

        tprint_success(f"✅ Generated {len(feature_columns)} basic features")
        return feature_columns

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _create_feature_cache_key(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """Create cache key for features."""
        data_hash = hash(str(market_data.shape) + str(market_data.index[0]) + str(market_data.index[-1]))
        return f"features_{config['symbol']}_{config['timeframe']}_{data_hash}"

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        metrics = self.cache_metrics.copy()

        if metrics['load_times']:
            metrics['avg_load_time'] = np.mean(metrics['load_times'])
        else:
            metrics['avg_load_time'] = 0.0

        if metrics['save_times']:
            metrics['avg_save_time'] = np.mean(metrics['save_times'])
        else:
            metrics['avg_save_time'] = 0.0

        total_operations = metrics['hits'] + metrics['misses']
        metrics['hit_rate'] = metrics['hits'] / total_operations if total_operations > 0 else 0.0

        return metrics

    def reset_cache_metrics(self):
        """Reset cache metrics."""
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0,
            'load_times': [],
            'save_times': []
        }

    async def store_klines_data(self, data: pd.DataFrame, symbol: str, exchange: str,
                               interval: str, batch_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store klines data using KlinesParquetManager for efficient storage.

        Args:
            data: Klines DataFrame with OHLCV data
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "binance")
            interval: Data interval (e.g., "1m")
            batch_id: Optional batch identifier
            metadata: Additional metadata to store

        Returns:
            True if storage was successful, False otherwise
        """
        try:
            tprint_debug(f"📦 Storing klines data for {symbol} on {exchange} ({interval})")

            # Validate data format
            if not self._validate_klines_data(data):
                tprint_error("❌ Invalid klines data format")
                return False

            # Store using KlinesParquetManager
            success = self.klines_manager.store_klines(
                data, symbol, exchange, interval, batch_id, metadata
            )

            if success:
                self.stats['klines_stores'] += 1
                tprint_success(f"✅ Stored {len(data)} klines records for {symbol}")
            else:
                self.stats['errors'] += 1
                tprint_error(f"❌ Failed to store klines data for {symbol}")

            return success

        except Exception as e:
            self.stats['errors'] += 1
            tprint_error(f"❌ Error storing klines data: {e}")
            return False

    async def load_klines_data(self, symbol: str, exchange: str, interval: str,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              batch_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load klines data using KlinesParquetManager.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            start_time: Optional start time filter
            end_time: Optional end time filter
            batch_id: Optional specific batch to load

        Returns:
            DataFrame containing klines data
        """
        try:
            tprint_debug(f"📥 Loading klines data for {symbol} on {exchange} ({interval})")

            # Load using KlinesParquetManager
            data = self.klines_manager.load_klines(
                symbol, exchange, interval, start_time, end_time, batch_id
            )

            if not data.empty:
                self.stats['klines_loads'] += 1
                tprint_success(f"✅ Loaded {len(data)} klines records for {symbol}")
            else:
                tprint_warning(f"⚠️ No klines data found for {symbol}")

            return data

        except Exception as e:
            self.stats['errors'] += 1
            tprint_error(f"❌ Error loading klines data: {e}")
            return pd.DataFrame()

    def _validate_klines_data(self, data: pd.DataFrame) -> bool:
        """Validate klines data format."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        if data is None or data.empty:
            return False

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            tprint_error(f"❌ Missing required columns: {missing_columns}")
            return False

        # Check for valid OHLCV data
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                tprint_error(f"❌ Column {col} is not numeric")
                return False

            if data[col].isnull().any():
                tprint_warning(f"⚠️ Column {col} contains null values")

        # Check OHLC relationships
        if not (data['high'] >= data['low']).all():
            tprint_error("❌ High prices must be >= low prices")
            return False

        if not (data['high'] >= data['open']).all():
            tprint_error("❌ High prices must be >= open prices")
            return False

        if not (data['high'] >= data['close']).all():
            tprint_error("❌ High prices must be >= close prices")
            return False

        if not (data['low'] <= data['open']).all():
            tprint_error("❌ Low prices must be <= open prices")
            return False

        if not (data['low'] <= data['close']).all():
            tprint_error("❌ Low prices must be <= close prices")
            return False

        return True

    def get_klines_storage_stats(self) -> Dict[str, Any]:
        """Get klines storage statistics."""
        try:
            return self.klines_manager.get_storage_stats()
        except Exception as e:
            tprint_error(f"❌ Error getting storage stats: {e}")
            return {"error": str(e)}

    def list_available_klines_data(self) -> List[Dict[str, Any]]:
        """List all available klines data."""
        try:
            return self.klines_manager.list_available_data()
        except Exception as e:
            tprint_error(f"❌ Error listing available data: {e}")
            return []

    async def update_klines_data(self, data: pd.DataFrame, symbol: str, exchange: str,
                                interval: str, append_mode: bool = True) -> bool:
        """
        Update existing klines data.

        Args:
            data: New klines data
            symbol: Trading symbol
            exchange: Exchange name
            interval: Data interval
            append_mode: If True, append to existing data; if False, replace

        Returns:
            True if update was successful, False otherwise
        """
        try:
            tprint_debug(f"🔄 Updating klines data for {symbol} on {exchange} ({interval})")

            # Validate data format
            if not self._validate_klines_data(data):
                tprint_error("❌ Invalid klines data format")
                return False

            # Update using KlinesParquetManager
            success = self.klines_manager.update_klines(
                data, symbol, exchange, interval, append_mode
            )

            if success:
                tprint_success(f"✅ Updated klines data for {symbol}")
            else:
                tprint_error(f"❌ Failed to update klines data for {symbol}")

            return success

        except Exception as e:
            self.stats['errors'] += 1
            tprint_error(f"❌ Error updating klines data: {e}")
            return False
        tprint_success("✅ Cache metrics reset")

    def _validate_provided_data(self, data: pd.DataFrame) -> bool:
        """Validate provided data for basic requirements."""
        try:
            # Check if data is not empty
            if data.empty:
                tprint_warning("⚠️ Provided data is empty")
                return False

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                tprint_warning(f"⚠️ Provided data missing required columns: {missing_columns}")
                return False

            # Check for numeric data types
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    tprint_warning(f"⚠️ Column {col} is not numeric in provided data")
                    return False

            # Check for reasonable data ranges
            if (data['high'] < data['low']).any():
                tprint_warning("⚠️ Invalid OHLC data: high < low detected")
                return False

            if (data['high'] < data['open']).any() or (data['high'] < data['close']).any():
                tprint_warning("⚠️ Invalid OHLC data: high < open/close detected")
                return False

            if (data['low'] > data['open']).any() or (data['low'] > data['close']).any():
                tprint_warning("⚠️ Invalid OHLC data: low > open/close detected")
                return False

            tprint_success("✅ Provided data validation passed")
            return True

        except Exception as e:
            tprint_warning(f"⚠️ Data validation error: {e}")
            return False

    async def _process_provided_data(self, data: pd.DataFrame,
                                   pipeline_state: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """Process provided data through the pipeline."""
        try:
            # Create a copy to avoid modifying original data
            processed_data = data.copy()

            # Apply data quality checks
            tprint_debug("🔍 Applying data quality checks to provided data")
            nan_analysis = analyze_nan_values_detailed(processed_data)
            quality_metrics = calculate_data_quality_metrics(processed_data)

            # Log quality metrics
            tprint_debug(f"📊 Data quality: {quality_metrics.get('missing_percentage', 0):.1f}% missing, "
                        f"{quality_metrics.get('duplicate_percentage', 0):.1f}% duplicates")

            # Define required columns for cleaning
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Apply basic cleaning if needed
            if quality_metrics.get('missing_percentage', 0) > 5.0:
                tprint_info("🧹 Applying basic data cleaning to provided data")
                processed_data = processed_data.dropna(subset=required_columns)

            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in processed_data.columns:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

            # Remove any remaining NaN values
            processed_data = processed_data.dropna()

            if processed_data.empty:
                tprint_warning("⚠️ All data removed during cleaning")
                return None

            tprint_success(f"✅ Processed provided data: {processed_data.shape}")
            return processed_data

        except Exception as e:
            tprint_error(f"❌ Error processing provided data: {e}")
            return None

    def _enhanced_label_merge(self, market_data: pd.DataFrame,
                            labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced label merging with better index handling and data preservation."""
        merge_stats = {
            'success': False,
            'rows_merged': 0,
            'columns_added': 0,
            'rows_dropped': 0,
            'index_strategy': 'unknown',
            'error': None
        }

        try:
            # Strategy 1: Try exact index match first
            common_index = market_data.index.intersection(labels_df.index)
            if len(common_index) > 0:
                merge_stats['index_strategy'] = 'exact_match'
                merge_stats['rows_merged'] = len(common_index)

                # Use the common index
                merged_data = market_data.loc[common_index].copy()
                labels_to_merge = labels_df.loc[common_index]

                # Add label columns
                for col in labels_to_merge.columns:
                    if col not in merged_data.columns:
                        merged_data[col] = labels_to_merge[col]
                        merge_stats['columns_added'] += 1

                merge_stats['success'] = True
                tprint_success(f"✅ Exact index match: {len(common_index)} rows merged")
                return merged_data, merge_stats

            # Strategy 2: Try time-based alignment if both have datetime indices
            if (isinstance(market_data.index, pd.DatetimeIndex) and
                isinstance(labels_df.index, pd.DatetimeIndex)):

                merge_stats['index_strategy'] = 'time_alignment'

                # Find overlapping time range
                market_start, market_end = market_data.index.min(), market_data.index.max()
                labels_start, labels_end = labels_df.index.min(), labels_df.index.max()

                overlap_start = max(market_start, labels_start)
                overlap_end = min(market_end, labels_end)

                if overlap_start < overlap_end:
                    # Filter to overlapping time range
                    market_filtered = market_data[(market_data.index >= overlap_start) &
                                                (market_data.index <= overlap_end)]
                    labels_filtered = labels_df[(labels_df.index >= overlap_start) &
                                             (labels_df.index <= overlap_end)]

                    # Use nearest time alignment
                    merged_data = market_filtered.copy()

                    for col in labels_filtered.columns:
                        if col not in merged_data.columns:
                            # Use forward fill for time alignment
                            aligned_labels = labels_filtered[col].reindex(
                                merged_data.index, method='ffill'
                            )
                            merged_data[col] = aligned_labels
                            merge_stats['columns_added'] += 1

                    merge_stats['rows_merged'] = len(merged_data)
                    merge_stats['success'] = True
                    tprint_success(f"✅ Time alignment: {len(merged_data)} rows merged")
                    return merged_data, merge_stats

            # Strategy 3: Try positional alignment as last resort
            if len(market_data) == len(labels_df):
                merge_stats['index_strategy'] = 'positional'
                merge_stats['rows_merged'] = len(market_data)

                merged_data = market_data.copy()
                for col in labels_df.columns:
                    if col not in merged_data.columns:
                        merged_data[col] = labels_df[col].values
                        merge_stats['columns_added'] += 1

                merge_stats['success'] = True
                tprint_warning("⚠️ Using positional alignment - verify data correctness")
                return merged_data, merge_stats

            # If all strategies fail
            merge_stats['error'] = "No compatible alignment strategy found"
            tprint_warning("⚠️ No compatible alignment strategy found for label merging")
            return market_data, merge_stats

        except Exception as e:
            merge_stats['error'] = str(e)
            tprint_error(f"❌ Enhanced label merge failed: {e}")
            return market_data, merge_stats
