#!/usr/bin/env python3
"""Real-time Streaming Pipeline for Regime Discovery.

This module implements a real-time streaming pipeline for processing live market data
with an agnostic approach that can work with different exchanges (Binance, Gate.io, etc.).
"""

import ast
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Protocol
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

class DataStream(Protocol):
    """Protocol for data streaming interfaces."""
    
    async def connect(self) -> None:
        """Connect to data stream."""
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from data stream."""
        ...
    
    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to market data stream."""
        ...
    
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data from stream."""
        ...

class ExchangeDataStream(ABC):
    """Abstract base class for exchange-specific data streams."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.subscriptions = {}
        
    @abstractmethod
    async def connect(self) -> None:
        """Connect to exchange data stream."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange data stream."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to market data for symbol and timeframe."""
        pass
    
    @abstractmethod
    async def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest market data."""
        pass
    
    @abstractmethod
    def parse_data(self, raw_data: Any) -> pd.DataFrame:
        """Parse raw exchange data into standardized format."""
        pass

class BinanceDataStream(ExchangeDataStream):
    """Binance-specific data stream implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'wss://stream.binance.com:9443/ws/')
        
    async def connect(self) -> None:
        """Connect to Binance WebSocket stream."""
        # Implementation would use websockets library
        # For now, simulate connection
        self.is_connected = True
        print("🔗 Connected to Binance WebSocket stream")
        
    async def disconnect(self) -> None:
        """Disconnect from Binance stream."""
        self.is_connected = False
        print("🔌 Disconnected from Binance stream")
        
    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to Binance kline stream."""
        stream_name = f"{symbol.lower()}@kline_{timeframe}"
        self.subscriptions[stream_name] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'subscribed_at': datetime.now()
        }
        print(f"📡 Subscribed to {symbol} {timeframe} on Binance")
        
    async def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data from Binance stream."""
        if not self.is_connected:
            return None
        
        # Simulate data reception
        # In real implementation, this would read from WebSocket
        return {
            'symbol': 'BTCUSDT',
            'timestamp': datetime.now(),
            'open': 50000.0 + np.random.normal(0, 100),
            'high': 50100.0 + np.random.normal(0, 100),
            'low': 49900.0 + np.random.normal(0, 100),
            'close': 50050.0 + np.random.normal(0, 100),
            'volume': 1000.0 + np.random.normal(0, 100)
        }
    
    def parse_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Parse Binance data into standardized format."""
        return pd.DataFrame([{
            'timestamp': raw_data['timestamp'],
            'open': raw_data['open'],
            'high': raw_data['high'],
            'low': raw_data['low'],
            'close': raw_data['close'],
            'volume': raw_data['volume']
        }])

class GateIODataStream(ExchangeDataStream):
    """Gate.io-specific data stream implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'wss://api.gateio.ws/ws/v4/')
        
    async def connect(self) -> None:
        """Connect to Gate.io WebSocket stream."""
        self.is_connected = True
        print("🔗 Connected to Gate.io WebSocket stream")
        
    async def disconnect(self) -> None:
        """Disconnect from Gate.io stream."""
        self.is_connected = False
        print("🔌 Disconnected from Gate.io stream")
        
    async def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to Gate.io kline stream."""
        stream_name = f"{symbol}_{timeframe}"
        self.subscriptions[stream_name] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'subscribed_at': datetime.now()
        }
        print(f"📡 Subscribed to {symbol} {timeframe} on Gate.io")
        
    async def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest data from Gate.io stream."""
        if not self.is_connected:
            return None
        
        # Simulate data reception
        return {
            'symbol': 'BTC_USDT',
            'timestamp': datetime.now(),
            'open': 50000.0 + np.random.normal(0, 100),
            'high': 50100.0 + np.random.normal(0, 100),
            'low': 49900.0 + np.random.normal(0, 100),
            'close': 50050.0 + np.random.normal(0, 100),
            'volume': 1000.0 + np.random.normal(0, 100)
        }
    
    def parse_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Parse Gate.io data into standardized format."""
        return pd.DataFrame([{
            'timestamp': raw_data['timestamp'],
            'open': raw_data['open'],
            'high': raw_data['high'],
            'low': raw_data['low'],
            'close': raw_data['close'],
            'volume': raw_data['volume']
        }])

class DataStreamFactory:
    """Factory for creating exchange-specific data streams."""
    
    @staticmethod
    def create_stream(exchange: str, config: Dict[str, Any]) -> ExchangeDataStream:
        """Create data stream for specified exchange."""
        if exchange.lower() == 'binance':
            return BinanceDataStream(config)
        elif exchange.lower() == 'gateio':
            return GateIODataStream(config)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    exchange: str
    symbol: str
    timeframe: str
    buffer_size: int = 1000
    processing_interval: float = 1.0  # seconds
    regime_update_interval: int = 100  # update regime model every N samples
    min_samples_for_regime: int = 500
    enable_persistence: bool = True
    enable_forecasting: bool = True

class RealTimeRegimeProcessor:
    """Real-time regime processor for streaming data."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.data_buffer = []
        self.regime_buffer = []
        self.current_regime = None
        self.regime_model = None
        self.is_model_trained = False
        self.sample_count = 0
        self.last_regime_update = 0
        
        # Initialize regime discovery components
        self._initialize_regime_components()
        
    def _initialize_regime_components(self) -> None:
        """Initialize regime discovery components."""
        from .step03_streaming_regime_discovery import StreamingRegimeDiscovery
        from .step03_regime_persistence_forecasting import RegimePersistenceForecaster
        
        self.streaming_processor = StreamingRegimeDiscovery({
            'chunk_size': self.config.buffer_size,
            'update_frequency': self.config.regime_update_interval
        })
        
        if self.config.enable_persistence:
            self.forecaster = RegimePersistenceForecaster()
        else:
            self.forecaster = None
    
    async def process_data_point(self, data_point: pd.DataFrame) -> Dict[str, Any]:
        """Process a single data point in real-time."""
        self.sample_count += 1
        
        # Add to buffer
        self.data_buffer.append(data_point)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.config.buffer_size:
            self.data_buffer.pop(0)
        
        # Check if we have enough data for regime detection
        if len(self.data_buffer) < self.config.min_samples_for_regime:
            return {
                'regime': None,
                'confidence': 0.0,
                'forecast': None,
                'sample_count': self.sample_count,
                'status': 'insufficient_data'
            }
        
        # Update regime model if needed
        if self._should_update_regime_model():
            await self._update_regime_model()
        
        # Predict current regime
        current_regime = self._predict_current_regime()
        
        # Generate forecast if enabled
        forecast = None
        if self.config.enable_forecasting and self.forecaster and self.is_model_trained:
            forecast = await self._generate_forecast()
        
        return {
            'regime': current_regime,
            'confidence': self._calculate_regime_confidence(),
            'forecast': forecast,
            'sample_count': self.sample_count,
            'status': 'active',
            'buffer_size': len(self.data_buffer)
        }
    
    def _should_update_regime_model(self) -> bool:
        """Check if regime model should be updated."""
        return (
            self.sample_count - self.last_regime_update >= self.config.regime_update_interval and
            len(self.data_buffer) >= self.config.min_samples_for_regime
        )
    
    async def _update_regime_model(self) -> None:
        """Update the regime model with recent data."""
        try:
            # Convert buffer to DataFrame
            recent_data = pd.concat(self.data_buffer, ignore_index=True)
            
            # Update streaming processor
            data_iterator = self.streaming_processor.create_data_iterator(recent_data)
            
            # Process recent data
            for chunk_result in self.streaming_processor.process_data_stream(data_iterator):
                pass  # Process all chunks
            
            # Update regime model
            if self.streaming_processor.is_model_trained:
                self.regime_model = self.streaming_processor.regime_model
                self.is_model_trained = True
                
                # Build persistence models if enabled
                if self.config.enable_persistence and self.forecaster:
                    recent_regimes = self.streaming_processor.regime_buffer
                    if len(recent_regimes) > 100:
                        self.persistence_models = self.forecaster.build_persistence_models(
                            recent_data, np.array(list(recent_regimes))
                        )
            
            self.last_regime_update = self.sample_count
            print(f"🔄 Updated regime model at sample {self.sample_count}")
            
        except Exception as e:
            print(f"⚠️ Error updating regime model: {e}")
    
    def _predict_current_regime(self) -> Optional[int]:
        """Predict current regime using trained model."""
        if not self.is_model_trained or not self.data_buffer:
            return None
        
        try:
            # Get latest data point
            latest_data = self.data_buffer[-1]
            
            # Extract features
            features = self._extract_realtime_features(latest_data)
            
            # Predict regime
            if self.regime_model:
                regime = self.regime_model.predict(features.reshape(1, -1))[0]
                return int(regime)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error predicting regime: {e}")
            return None
    
    def _extract_realtime_features(self, data_point: pd.DataFrame) -> np.ndarray:
        """Extract features from a single data point."""
        features = []
        
        # Price-based features
        if 'close' in data_point.columns:
            close_price = data_point['close'].iloc[0]
            features.append(close_price)
            
            # Calculate returns if we have previous data
            if len(self.data_buffer) > 1:
                prev_close = self.data_buffer[-2]['close'].iloc[0]
                returns = (close_price - prev_close) / prev_close
                features.append(returns)
            else:
                features.append(0.0)
        
        # Volume-based features
        if 'volume' in data_point.columns:
            volume = data_point['volume'].iloc[0]
            features.append(volume)
            
            # Volume ratio if we have previous data
            if len(self.data_buffer) > 1:
                prev_volume = self.data_buffer[-2]['volume'].iloc[0]
                volume_ratio = volume / prev_volume if prev_volume > 0 else 1.0
                features.append(volume_ratio)
            else:
                features.append(1.0)
        
        # Volatility features
        if 'high' in data_point.columns and 'low' in data_point.columns:
            high = data_point['high'].iloc[0]
            low = data_point['low'].iloc[0]
            close = data_point['close'].iloc[0]
            
            if close > 0:
                volatility = (high - low) / close
                features.append(volatility)
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def _calculate_regime_confidence(self) -> float:
        """Calculate confidence in current regime prediction."""
        if not self.is_model_trained:
            return 0.0
        
        # Simple confidence based on model training status and buffer size
        buffer_confidence = min(len(self.data_buffer) / self.config.min_samples_for_regime, 1.0)
        model_confidence = 0.8 if self.is_model_trained else 0.0
        
        return (buffer_confidence + model_confidence) / 2.0
    
    async def _generate_forecast(self) -> Optional[Dict[str, Any]]:
        """Generate regime forecast."""
        if not self.forecaster or not hasattr(self, 'persistence_models'):
            return None
        
        try:
            # Get current regime
            current_regime = self._predict_current_regime()
            if current_regime is None:
                return None
            
            # Get latest data
            latest_data = self.data_buffer[-1]
            
            # Generate forecast
            forecast = self.forecaster.forecast_regime_transitions(
                latest_data, current_regime, self.persistence_models
            )
            
            return forecast
            
        except Exception as e:
            print(f"⚠️ Error generating forecast: {e}")
            return None

class RealTimeStreamingPipeline:
    """Main real-time streaming pipeline."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.data_stream = None
        self.regime_processor = None
        self.is_running = False
        self.callbacks = []
        
    async def initialize(self) -> None:
        """Initialize the streaming pipeline."""
        print("🚀 Initializing real-time streaming pipeline...")
        
        # Create data stream
        self.data_stream = DataStreamFactory.create_stream(
            self.config.exchange, 
            {'api_key': None, 'secret_key': None}  # Would be loaded from config
        )
        
        # Initialize regime processor
        self.regime_processor = RealTimeRegimeProcessor(self.config)
        
        # Connect to data stream
        await self.data_stream.connect()
        
        # Subscribe to market data
        await self.data_stream.subscribe(self.config.symbol, self.config.timeframe)
        
        print("✅ Streaming pipeline initialized successfully")
    
    async def start_streaming(self) -> None:
        """Start the real-time streaming process."""
        if self.is_running:
            print("⚠️ Streaming already running")
            return
        
        self.is_running = True
        print(f"📡 Starting real-time streaming for {self.config.symbol} on {self.config.exchange}")
        
        try:
            while self.is_running:
                # Get latest data
                raw_data = await self.data_stream.get_latest_data()
                
                if raw_data:
                    # Parse data
                    data_point = self.data_stream.parse_data(raw_data)
                    
                    # Process data point
                    result = await self.regime_processor.process_data_point(data_point)
                    
                    # Notify callbacks
                    await self._notify_callbacks(result)
                    
                    # Print status periodically
                    if result['sample_count'] % 100 == 0:
                        print(f"📊 Processed {result['sample_count']} samples, "
                              f"Current regime: {result['regime']}, "
                              f"Confidence: {result['confidence']:.2f}")
                
                # Wait before next iteration
                await asyncio.sleep(self.config.processing_interval)
                
        except Exception as e:
            print(f"❌ Error in streaming pipeline: {e}")
        finally:
            self.is_running = False
    
    async def stop_streaming(self) -> None:
        """Stop the streaming process."""
        print("🛑 Stopping streaming pipeline...")
        self.is_running = False
        
        if self.data_stream:
            await self.data_stream.disconnect()
        
        print("✅ Streaming pipeline stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for regime updates."""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, result: Dict[str, Any]) -> None:
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                print(f"⚠️ Error in callback: {e}")

class StreamingRegimeDiscovery:
    """Main interface for real-time streaming regime discovery."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        
    async def start_realtime_discovery(self, exchange: str, symbol: str, timeframe: str) -> None:
        """Start real-time regime discovery."""
        streaming_config = StreamingConfig(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            buffer_size=self.config.get('buffer_size', 1000),
            processing_interval=self.config.get('processing_interval', 1.0),
            regime_update_interval=self.config.get('regime_update_interval', 100),
            min_samples_for_regime=self.config.get('min_samples_for_regime', 500),
            enable_persistence=self.config.get('enable_persistence', True),
            enable_forecasting=self.config.get('enable_forecasting', True)
        )
        
        self.pipeline = RealTimeStreamingPipeline(streaming_config)
        
        # Add callback for regime updates
        self.pipeline.add_callback(self._on_regime_update)
        
        # Initialize and start
        await self.pipeline.initialize()
        await self.pipeline.start_streaming()
    
    async def stop_realtime_discovery(self) -> None:
        """Stop real-time regime discovery."""
        if self.pipeline:
            await self.pipeline.stop_streaming()
    
    async def _on_regime_update(self, result: Dict[str, Any]) -> None:
        """Handle regime update callback."""
        # This can be customized based on requirements
        if result['regime'] is not None:
            print(f"🔄 Regime update: {result['regime']} (confidence: {result['confidence']:.2f})")
            
            if result['forecast']:
                forecast = result['forecast']
                print(f"🔮 Forecast: {forecast.get('combined_forecast', {}).get('forecast', 'unknown')}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current streaming status."""
        if not self.pipeline or not self.pipeline.regime_processor:
            return {'status': 'not_initialized'}
        
        processor = self.pipeline.regime_processor
        return {
            'status': 'running' if self.pipeline.is_running else 'stopped',
            'sample_count': processor.sample_count,
            'buffer_size': len(processor.data_buffer),
            'is_model_trained': processor.is_model_trained,
            'current_regime': processor.current_regime,
            'last_regime_update': processor.last_regime_update
        }