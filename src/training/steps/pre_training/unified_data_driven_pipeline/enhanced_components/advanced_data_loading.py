"""
Advanced Data Loading and Management for Unified Data-Driven Pipeline.

This module provides comprehensive data loading infrastructure similar to
FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import asyncio
import logging
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

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

import numpy as np
import pandas as pd

# Try to import feature cache service
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FEATURE_CACHE_AVAILABLE = False
    tprint_warning("⚠️ FeatureCacheService not available")

# Try to import ares launcher integration
try:
    from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import AresLauncherFeatureLookbackOptimizer
    ARES_LAUNCHER_AVAILABLE = True
except ImportError:
    ARES_LAUNCHER_AVAILABLE = False
    tprint_warning("⚠️ AresLauncherFeatureLookbackOptimizer not available")


class AdvancedDataLoader:
    """
    Advanced data loader for unified pipeline.
    
    Provides comprehensive data loading, caching, and management capabilities
    similar to FeatureLookbackOptimizationComponent.
    """

    def __init__(self, logger=None):
        """Initialize the advanced data loader."""
        self.logger = logger or logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Initialize feature cache if available
        if FEATURE_CACHE_AVAILABLE:
            self.feature_cache = FeatureCacheService(subdirectory="unified_pipeline")
            tprint_success("✅ Feature cache initialized")
        else:
            self.feature_cache = None
            tprint_warning("⚠️ Feature cache not available")

        # Initialize ares launcher integration if available
        if ARES_LAUNCHER_AVAILABLE:
            self.ares_integration = AresLauncherFeatureLookbackOptimizer()
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
        
        # If data is already provided and not empty, use it
        if data is not None and not data.empty:
            tprint_success(f"✅ Using provided data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data

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
        
        # Add labeling data if available using safe operations
        if labeling_data and isinstance(labeling_data, dict):
            labels_df = labeling_data.get('labeled_data')
            if labels_df is not None and isinstance(labels_df, pd.DataFrame):
                # Use safe merge operation
                try:
                    # Find common index
                    common_index = optimization_data.index.intersection(labels_df.index)
                    if len(common_index) > 0:
                        optimization_data = optimization_data.loc[common_index]
                        labels_to_merge = labels_df.loc[common_index]
                        
                        # Add label columns safely
                        for col in labels_to_merge.columns:
                            if col not in optimization_data.columns:
                                optimization_data[col] = labels_to_merge[col]
                        
                        tprint_success(f"✅ Merged {len(labels_to_merge.columns)} label columns")
                    else:
                        tprint_warning("⚠️ No common index found for label merging")
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
        
        # Generate synthetic OHLCV data
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
        tprint_success("✅ Cache metrics reset")