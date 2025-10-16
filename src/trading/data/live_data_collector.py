"""
Live Data Collector for Real-Time Trading Analysis

This module provides a comprehensive live data collection system that fetches
market data every 30 seconds and integrates with ML models for real-time analysis.
Enhanced version with multi-timeframe support and ML integration.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel
from src.utils.enhanced_error_handler import get_enhanced_error_handler
from src.utils.memory_management.streaming_data_processor import get_streaming_processor, with_memory_optimization
from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager
from src.utils.data.processing.data_processing import optimize_dataframe_dtypes
from src.exchange.binance import BinanceExchange
from src.config.config import get_static_config

logger = system_logger.getChild('LiveDataCollector')

class CollectionMode(Enum):
    """Data collection modes."""
    LIVE = "live"      # Real-time data every 30s
    SIMULATED = "simulated"  # Historical data replay for testing
    HYBRID = "hybrid"   # Mix of live and cached data

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"       # Full validation and processing
    MEDIUM = "medium"   # Basic validation only
    LOW = "low"         # Minimal processing for speed

class CollectionInterval(Enum):
    """Collection interval modes."""
    HMM = 15 * 60      # 15 minutes - for HMM regime detection
    ANALYST = 2 * 60   # 2 minutes - for Analyst trade decisions
    TACTICIAN = 30     # 30 seconds - for Tactician timing decisions
    FAST = 15          # 15 seconds - for high-frequency trading
    STANDARD = 30      # 30 seconds - standard live trading

@dataclass
class LiveDataConfig:
    """Configuration for live data collection."""
    symbol: str = "ETH"
    exchange: str = "binance"
    interval: CollectionInterval = CollectionInterval.STANDARD
    collection_mode: CollectionMode = CollectionMode.LIVE
    quality_level: DataQuality = DataQuality.HIGH
    buffer_size: int = 1000  # Keep last N candles
    enable_ml_predictions: bool = True
    ml_model_path: Optional[str] = None
    feature_engineering: bool = True
    real_time_validation: bool = True
    error_recovery: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def interval_seconds(self) -> int:
        """Get the interval in seconds."""
        return self.interval.value

@dataclass
class LiveDataPoint:
    """Represents a single live data point with ML analysis."""
    timestamp: datetime
    symbol: str
    exchange: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    ml_predictions: Optional[Dict[str, Any]] = None
    quality_score: float = 1.0
    processing_time_ms: float = 0.0
    collection_metadata: Dict[str, Any] = field(default_factory=dict)

class LiveDataCollector:
    """
    Live Data Collector for Real-Time Trading Analysis

    Fetches market data every 15 or 30 seconds and integrates with ML models
    for real-time analysis and prediction. Defaults to ETH symbol.
    """

    def __init__(self, config: LiveDataConfig):
        self.config = config
        self.logger = logger.getChild(f'{config.symbol}_{config.exchange}')

        # Core components
        self.exchange_client = None
        self.ml_model = None
        self.feature_engineer = None

        # Data buffers for different timeframes
        self.data_buffer: List[LiveDataPoint] = []
        self.processed_buffer: List[Dict[str, Any]] = []
        self.hmm_buffer: List[LiveDataPoint] = []  # 1h data for HMM
        self.analyst_buffer: List[LiveDataPoint] = []  # 5m data for Analyst
        self.tactician_buffer: List[LiveDataPoint] = []  # 1m data for Tactician

        # State management
        self.is_running = False
        self.last_collection_time: Optional[datetime] = None
        self.collection_count = 0
        self.error_count = 0

        # Callbacks
        self.on_data_callbacks: List[Callable[[LiveDataPoint], None]] = []
        self.on_error_callbacks: List[Callable[[Exception], None]] = []

        # Advanced systems
        self.error_recovery = get_enhanced_error_handler() if config.error_recovery else None
        self.streaming_processor = get_streaming_processor()
        self.hardware_manager = get_unified_hardware_manager()

        # Performance optimization
        self.memory_optimized = True

        # Note: Components will be initialized asynchronously in start_collection()

    async def _initialize_components(self):
        """Initialize exchange client and ML components."""
        try:
            # Initialize exchange client (skip for simulated mode)
            if self.config.collection_mode != CollectionMode.SIMULATED:
                tprint_info(f"🔄 Initializing {self.config.exchange} exchange client...")

                # Use Binance exchange directly
                if self.config.exchange.lower() == "binance":
                    self.exchange_client = BinanceExchange()
                else:
                    tprint_warning(f"⚠️ Exchange {self.config.exchange} not supported, falling back to Binance")
                    self.exchange_client = BinanceExchange()

                # Initialize the exchange client
                if self.exchange_client:
                    success = await self.exchange_client.initialize()
                    if not success:
                        tprint_warning("⚠️ Failed to initialize exchange client, continuing in simulated mode")
                        self.logger.warning("⚠️ Failed to initialize exchange client, continuing in simulated mode")
                        self.config.collection_mode = CollectionMode.SIMULATED
                        self.exchange_client = None
                    else:
                        tprint_success(f"✅ {self.config.exchange} exchange client initialized")
                else:
                    tprint_warning("⚠️ Could not create exchange client, falling back to simulated mode")
                    self.logger.warning("⚠️ Could not create exchange client, falling back to simulated mode")
                    self.config.collection_mode = CollectionMode.SIMULATED
            else:
                tprint_info("🔄 Simulated mode: Skipping exchange client initialization")
                self.logger.info("🔄 Simulated mode: Skipping exchange client initialization")

            # Initialize ML model if enabled
            if self.config.enable_ml_predictions and self.config.ml_model_path:
                tprint_info("🤖 Loading ML model for predictions...")
                self._load_ml_model()

            # Initialize feature engineering
            if self.config.feature_engineering:
                tprint_info("⚙️ Initializing feature engineering...")
                self._initialize_feature_engineering()

            # Initialize hardware optimization
            if self.hardware_manager:
                tprint_info("🚀 Optimizing hardware performance...")
                await self.hardware_manager.initialize()
                await self.hardware_manager.optimize_for_trading()

            tprint_success(f"✅ Live data collector initialized (mode: {self.config.collection_mode.value}, interval: {self.config.interval_seconds}s)")
            self.logger.info(f"✅ Live data collector initialized (mode: {self.config.collection_mode.value}, interval: {self.config.interval_seconds}s)")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize components: {e}")
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def _load_ml_model(self):
        """Load ML model for real-time predictions."""
        try:
            import joblib
            from pathlib import Path

            model_path = Path(self.config.ml_model_path)
            if model_path.exists():
                self.ml_model = joblib.load(model_path)
                self.logger.info(f"✅ ML model loaded from {model_path}")
            else:
                self.logger.warning(f"⚠️ ML model not found at {model_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load ML model: {e}")

    def _initialize_feature_engineering(self):
        """Initialize real-time feature engineering."""
        try:
            # Import feature engineering components
            from src.feature_generation.utils.optimized_feature_orchestrator import OptimizedFeatureOrchestrator

            self.feature_engineer = OptimizedFeatureOrchestrator(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe="1m"  # Base timeframe
            )

            tprint_success("✅ Feature engineering initialized")
            self.logger.info("✅ Feature engineering initialized")

        except ImportError:
            # Try alternative feature engineering
            try:
                from src.feature_generation.utils.feature_generators import FeatureGenerators
                self.feature_engineer = FeatureGenerators()
                tprint_success("✅ Alternative feature engineering initialized")
            except ImportError:
                tprint_warning("⚠️ Feature engineering not available, using basic features")
                self.feature_engineer = None
        except Exception as e:
            tprint_warning(f"⚠️ Feature engineering not available: {e}")
            self.logger.warning(f"⚠️ Feature engineering not available: {e}")
            self.feature_engineer = None

    @handles_errors
    async def start_collection(self) -> bool:
        """Start live data collection."""
        if self.is_running:
            self.logger.warning("⚠️ Data collection already running")
            return False

        try:
            # Initialize components first
            await self._initialize_components()

            self.is_running = True
            self.logger.info(f"🚀 Starting live data collection for {self.config.symbol}")

            # Start collection loop
            asyncio.create_task(self._collection_loop())

            return True

        except Exception as e:
            self.is_running = False
            self.logger.error(f"❌ Failed to start collection: {e}")
            return False

    @handles_errors
    async def stop_collection(self) -> bool:
        """Stop live data collection."""
        if not self.is_running:
            return True

        self.is_running = False
        self.logger.info("🛑 Stopping live data collection")

        # Wait for cleanup
        await asyncio.sleep(0.1)
        return True

    async def _collection_loop(self):
        """Main data collection loop."""
        while self.is_running:
            try:
                start_time = time.time()

                # Collect data
                await self._collect_data_point()

                # Calculate sleep time to maintain 30s intervals
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.interval_seconds - elapsed)

                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.error_count += 1
                self.logger.error(f"❌ Collection error: {e}")
                await self._handle_error(e)

                # Brief pause on error
                await asyncio.sleep(5)

    @with_memory_optimization(chunk_size=1, max_memory_mb=512)
    async def _collect_data_point(self):
        """Collect a single data point."""
        try:
            collection_start = time.time()

            # Get latest kline data
            raw_data = await self._fetch_latest_data()

            if not raw_data:
                return

            # Create data point
            data_point = LiveDataPoint(
                timestamp=datetime.now(),
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                raw_data=raw_data,
                collection_metadata={
                    'collection_mode': self.config.collection_mode.value,
                    'quality_level': self.config.quality_level.value
                }
            )

            # Process data
            if self.config.quality_level != DataQuality.LOW:
                await self._process_data_point(data_point)

            # Add ML predictions
            if self.config.enable_ml_predictions and self.ml_model:
                await self._add_ml_predictions(data_point)

            # Update buffers
            self._update_buffers(data_point)

            # Optimize memory usage
            if self.memory_optimized and len(self.data_buffer) % 100 == 0:
                await self._optimize_memory_usage()

            # Record timing
            data_point.processing_time_ms = (time.time() - collection_start) * 1000

            # Update state
            self.last_collection_time = data_point.timestamp
            self.collection_count += 1

            # Trigger callbacks
            await self._trigger_callbacks(data_point)

            # Log progress
            if self.collection_count % 10 == 0:  # Log every 10 collections
                self.logger.info(f"📊 Collected {self.collection_count} data points, "
                               f"buffer size: {len(self.data_buffer)}")

        except Exception as e:
            self.logger.error(f"❌ Data point collection failed: {e}")
            raise

    async def _fetch_latest_data(self) -> Optional[Dict[str, Any]]:
        """Fetch latest market data."""
        try:
            if self.config.collection_mode == CollectionMode.LIVE:
                # Get latest 1m kline
                klines = await self.exchange_client.get_klines(
                    symbol=self.config.symbol,
                    interval="1m",
                    limit=1
                )

                if klines:
                    # Convert to dict format
                    latest_kline = klines[0]
                    return {
                        'timestamp': latest_kline.timestamp,
                        'open': latest_kline.open,
                        'high': latest_kline.high,
                        'low': latest_kline.low,
                        'close': latest_kline.close,
                        'volume': latest_kline.volume,
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange
                    }

            elif self.config.collection_mode == CollectionMode.SIMULATED:
                # Use historical data for testing
                return await self._simulate_data_fetch()

        except Exception as e:
            self.logger.error(f"❌ Data fetch failed: {e}")
            return None

    async def _simulate_data_fetch(self) -> Optional[Dict[str, Any]]:
        """Simulate data fetch for testing."""
        # This would load historical data and replay it
        # For now, return mock data
        return {
            'timestamp': datetime.now(),
            'open': 50000.0 + np.random.normal(0, 100),
            'high': 50100.0 + np.random.normal(0, 50),
            'low': 49900.0 + np.random.normal(0, 50),
            'close': 50000.0 + np.random.normal(0, 100),
            'volume': 100.0 + np.random.normal(0, 20),
            'symbol': self.config.symbol,
            'exchange': self.config.exchange
        }

    async def _process_data_point(self, data_point: LiveDataPoint):
        """Process raw data point."""
        try:
            processed_data = dict(data_point.raw_data)

            # Add basic features
            processed_data.update({
                'returns': self._calculate_returns(processed_data),
                'volatility': self._calculate_volatility(),
                'volume_ma': self._calculate_volume_ma(),
            })

            # Advanced feature engineering
            if self.feature_engineer and self.config.feature_engineering:
                features = await self._engineer_features(processed_data)
                processed_data.update(features)

            data_point.processed_data = processed_data

        except Exception as e:
            self.logger.warning(f"⚠️ Data processing failed: {e}")

    def _calculate_returns(self, data: Dict[str, Any]) -> float:
        """Calculate price returns."""
        try:
            if len(self.data_buffer) > 0:
                prev_close = self.data_buffer[-1].raw_data.get('close', data['close'])
                return (data['close'] - prev_close) / prev_close
            return 0.0
        except:
            return 0.0

    def _calculate_volatility(self) -> float:
        """Calculate rolling volatility."""
        try:
            if len(self.data_buffer) >= 20:
                closes = [dp.raw_data['close'] for dp in self.data_buffer[-20:]]
                returns = np.diff(closes) / closes[:-1]
                return np.std(returns)
            return 0.0
        except:
            return 0.0

    def _calculate_volume_ma(self) -> float:
        """Calculate volume moving average."""
        try:
            if len(self.data_buffer) >= 10:
                volumes = [dp.raw_data['volume'] for dp in self.data_buffer[-10:]]
                return np.mean(volumes)
            return 0.0
        except:
            return 0.0

    async def _engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer advanced features using existing system."""
        try:
            # Create DataFrame for feature engineering
            df = pd.DataFrame([data])

            # Use your existing feature engineering
            if hasattr(self.feature_engineer, 'engineer_features'):
                features_df = await self.feature_engineer.engineer_features(df)
                return features_df.iloc[0].to_dict()

            return {}

        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineering failed: {e}")
            return {}

    async def _add_ml_predictions(self, data_point: LiveDataPoint):
        """Add ML model predictions."""
        try:
            if not self.ml_model or not data_point.processed_data:
                return

            # Prepare features for ML model
            features = self._prepare_ml_features(data_point.processed_data)

            # Make prediction
            prediction = self.ml_model.predict([features])[0]

            # Get prediction probabilities if available
            probabilities = None
            if hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba([features])[0]

            data_point.ml_predictions = {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': max(probabilities) if probabilities is not None else None,
                'model_type': type(self.ml_model).__name__
            }

        except Exception as e:
            self.logger.warning(f"⚠️ ML prediction failed: {e}")

    def _prepare_ml_features(self, data: Dict[str, Any]) -> List[float]:
        """Prepare features for ML model input."""
        # This should match your ML model training features
        # Adjust based on your specific model requirements
        features = [
            data.get('close', 0),
            data.get('volume', 0),
            data.get('returns', 0),
            data.get('volatility', 0),
            data.get('volume_ma', 0),
        ]

        # Add any additional features your model expects
        return features

    def _update_buffers(self, data_point: LiveDataPoint):
        """Update data buffers for different timeframes."""
        self.data_buffer.append(data_point)

        # Maintain buffer size
        if len(self.data_buffer) > self.config.buffer_size:
            self.data_buffer.pop(0)

        # Update processed buffer
        if data_point.processed_data:
            self.processed_buffer.append(data_point.processed_data)
            if len(self.processed_buffer) > self.config.buffer_size:
                self.processed_buffer.pop(0)

        # Update timeframe-specific buffers
        self._update_timeframe_buffers(data_point)

    def _update_timeframe_buffers(self, data_point: LiveDataPoint):
        """Update buffers for different model timeframes."""
        # HMM buffer (1h data) - add every 60th data point
        if self.collection_count % 60 == 0:
            self.hmm_buffer.append(data_point)
            if len(self.hmm_buffer) > 100:  # Keep last 100 hours
                self.hmm_buffer.pop(0)

        # Analyst buffer (5m data) - add every 5th data point
        if self.collection_count % 5 == 0:
            self.analyst_buffer.append(data_point)
            if len(self.analyst_buffer) > 200:  # Keep last 200 5-min periods
                self.analyst_buffer.pop(0)

        # Tactician buffer (1m data) - add every data point
        self.tactician_buffer.append(data_point)
        if len(self.tactician_buffer) > 1000:  # Keep last 1000 minutes
            self.tactician_buffer.pop(0)

    async def _trigger_callbacks(self, data_point: LiveDataPoint):
        """Trigger data callbacks."""
        for callback in self.on_data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_point)
                else:
                    callback(data_point)
            except Exception as e:
                self.logger.warning(f"⚠️ Callback failed: {e}")

    async def _handle_error(self, error: Exception):
        """Handle collection errors."""
        for callback in self.on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                self.logger.warning(f"⚠️ Error callback failed: {e}")

    def add_data_callback(self, callback: Callable[[LiveDataPoint], None]):
        """Add a callback for new data points."""
        self.on_data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add a callback for errors."""
        self.on_error_callbacks.append(callback)

    def get_recent_data(self, n: int = 100) -> List[LiveDataPoint]:
        """Get the most recent n data points."""
        return self.data_buffer[-n:] if len(self.data_buffer) >= n else self.data_buffer.copy()

    def get_processed_data_df(self, n: int = 100) -> pd.DataFrame:
        """Get recent processed data as DataFrame."""
        recent_data = self.processed_buffer[-n:] if len(self.processed_buffer) >= n else self.processed_buffer
        return pd.DataFrame(recent_data)

    def get_timeframe_data(self, timeframe: str, n: int = 100) -> List[LiveDataPoint]:
        """Get data for specific timeframe."""
        if timeframe == "hmm":
            return self.hmm_buffer[-n:] if len(self.hmm_buffer) >= n else self.hmm_buffer.copy()
        elif timeframe == "analyst":
            return self.analyst_buffer[-n:] if len(self.analyst_buffer) >= n else self.analyst_buffer.copy()
        elif timeframe == "tactician":
            return self.tactician_buffer[-n:] if len(self.tactician_buffer) >= n else self.tactician_buffer.copy()
        else:
            return self.get_recent_data(n)

    def get_timeframe_dataframe(self, timeframe: str, n: int = 100) -> pd.DataFrame:
        """Get timeframe data as DataFrame."""
        data_points = self.get_timeframe_data(timeframe, n)
        if not data_points:
            return pd.DataFrame()

        # Convert to DataFrame
        data_list = []
        for dp in data_points:
            if dp.processed_data:
                data_list.append(dp.processed_data)

        df = pd.DataFrame(data_list)

        # Optimize memory usage for the returned DataFrame
        if self.memory_optimized and not df.empty:
            try:
                df = optimize_dataframe_dtypes(df)
            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed: {e}")

        return df

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'is_running': self.is_running,
            'collection_count': self.collection_count,
            'error_count': self.error_count,
            'buffer_size': len(self.data_buffer),
            'hmm_buffer_size': len(self.hmm_buffer),
            'analyst_buffer_size': len(self.analyst_buffer),
            'tactician_buffer_size': len(self.tactician_buffer),
            'last_collection_time': self.last_collection_time,
            'avg_processing_time_ms': np.mean([dp.processing_time_ms for dp in self.data_buffer[-100:]]) if self.data_buffer else 0,
            'ml_predictions_enabled': self.ml_model is not None,
            'feature_engineering_enabled': self.feature_engineer is not None
        }

    async def _optimize_memory_usage(self):
        """Optimize memory usage of data buffers."""
        try:
            if self.hardware_manager:
                # Use hardware manager for memory optimization
                await self.hardware_manager.optimize_memory_usage()

            # Convert processed buffer to optimized DataFrame and back
            if self.processed_buffer:
                temp_df = pd.DataFrame(self.processed_buffer[-100:])  # Keep last 100
                optimized_df = optimize_dataframe_dtypes(temp_df)
                self.processed_buffer = optimized_df.to_dict('records')

            tprint_info("🧹 Memory optimization completed")

        except Exception as e:
            tprint_warning(f"⚠️ Memory optimization failed: {e}")

# Convenience functions

def create_live_data_collector(
    symbol: str = "ETH",
    exchange: str = "binance",
    interval: CollectionInterval = CollectionInterval.STANDARD,
    enable_ml: bool = True,
    ml_model_path: Optional[str] = None
) -> LiveDataCollector:
    """Create a configured live data collector."""

    config = LiveDataConfig(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        enable_ml_predictions=enable_ml,
        ml_model_path=ml_model_path,
        feature_engineering=True,
        real_time_validation=True,
        error_recovery=True
    )

    return LiveDataCollector(config)

async def start_live_collection(
    symbol: str = "ETH",
    exchange: str = "binance",
    interval: CollectionInterval = CollectionInterval.STANDARD,
    ml_model_path: Optional[str] = None,
    data_callback: Optional[Callable] = None
) -> LiveDataCollector:
    """Start live data collection with default settings."""

    collector = create_live_data_collector(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        ml_model_path=ml_model_path
    )

    if data_callback:
        collector.add_data_callback(data_callback)

    success = await collector.start_collection()
    if success:
        print(f"✅ Live data collection started for {symbol} (every {interval.value}s)")
    else:
        print(f"❌ Failed to start live data collection")

    return collector
