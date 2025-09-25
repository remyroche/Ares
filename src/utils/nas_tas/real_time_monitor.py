"""
Real-Time Regime Monitoring System

This module provides real-time regime monitoring capabilities including streaming
data processing, regime change detection, and live performance tracking.
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import regime detection components
from .unified_regime_detector import UnifiedRegimeDetector, UnifiedRegimeResult
from .unified_regime_config import UnifiedRegimeConfig
from .performance_optimizer import get_performance_optimizer

logger = logging.getLogger(__name__)

@dataclass
class RegimeChangeEvent:
    """Represents a regime change event."""
    timestamp: datetime
    from_regime: int
    to_regime: int
    confidence: float
    economic_significance: float
    trading_viability: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RealTimeMetrics:
    """Real-time performance metrics."""
    timestamp: datetime
    current_regime: int
    regime_confidence: float
    economic_score: float
    trading_score: float
    stability_score: float
    processing_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_samples_per_second: float

class DataStreamProcessor:
    """Processes streaming market data for real-time regime detection."""
    
    def __init__(self, window_size: int = 100, overlap: int = 50):
        """Initialize data stream processor."""
        self.window_size = window_size
        self.overlap = overlap
        self.data_buffer = deque(maxlen=window_size * 2)
        self.timestamp_buffer = deque(maxlen=window_size * 2)
        
        tprint_info(f"📊 Data stream processor initialized: window={window_size}, overlap={overlap}")
    
    def add_data_point(self, data_point: Dict[str, float], timestamp: datetime):
        """Add a new data point to the stream."""
        # Convert data point to array format
        values = [data_point.get('open', 0), data_point.get('high', 0), 
                 data_point.get('low', 0), data_point.get('close', 0), 
                 data_point.get('volume', 0)]
        
        self.data_buffer.append(values)
        self.timestamp_buffer.append(timestamp)
        
        tprint_debug(f"📈 Data point added: {timestamp} - Close: {data_point.get('close', 0):.2f}")
    
    def get_processing_window(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the current processing window if it has enough data."""
        if len(self.data_buffer) < self.window_size:
            return None
        
        # Get the most recent window
        window_data = np.array(list(self.data_buffer)[-self.window_size:])
        window_timestamps = np.array(list(self.timestamp_buffer)[-self.window_size:])
        
        return window_data, window_timestamps
    
    def should_process(self) -> bool:
        """Check if we should process a new window."""
        if len(self.data_buffer) < self.window_size:
            return False
        
        # Process if we have new data beyond the overlap
        return len(self.data_buffer) >= self.window_size + self.overlap

class RegimeChangeDetector:
    """Detects regime changes in real-time."""
    
    def __init__(self, change_threshold: float = 0.3, confidence_threshold: float = 0.7):
        """Initialize regime change detector."""
        self.change_threshold = change_threshold
        self.confidence_threshold = confidence_threshold
        self.previous_regime = None
        self.previous_confidence = 0.0
        self.change_history = deque(maxlen=100)
        
        tprint_info(f"🔄 Regime change detector initialized: threshold={change_threshold}")
    
    def detect_change(self, current_regime: int, confidence: float, 
                     economic_significance: float, trading_viability: float,
                     timestamp: datetime) -> Optional[RegimeChangeEvent]:
        """Detect if a regime change has occurred."""
        change_detected = False
        
        # Check for regime change
        if self.previous_regime is not None:
            if (current_regime != self.previous_regime and 
                confidence >= self.confidence_threshold):
                change_detected = True
        
        # Check for confidence drop (potential regime instability)
        elif (self.previous_confidence - confidence) > self.change_threshold:
            change_detected = True
        
        if change_detected:
            event = RegimeChangeEvent(
                timestamp=timestamp,
                from_regime=self.previous_regime or -1,
                to_regime=current_regime,
                confidence=confidence,
                economic_significance=economic_significance,
                trading_viability=trading_viability,
                metadata={
                    'confidence_delta': confidence - self.previous_confidence,
                    'change_type': 'regime_shift' if self.previous_regime != current_regime else 'confidence_drop'
                }
            )
            
            self.change_history.append(event)
            
            tprint_warning(f"🔄 Regime change detected: {self.previous_regime} → {current_regime} (confidence: {confidence:.3f})")
            
            return event
        
        # Update previous values
        self.previous_regime = current_regime
        self.previous_confidence = confidence
        
        return None

class PerformanceMonitor:
    """Monitors real-time performance metrics."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """Initialize performance monitor."""
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.processed_samples = 0
        self.last_processing_time = time.time()
        
        tprint_info(f"📊 Performance monitor initialized: interval={monitoring_interval}s")
    
    def record_processing(self, result: UnifiedRegimeResult, processing_time: float):
        """Record processing metrics."""
        current_time = time.time()
        
        # Calculate throughput
        time_delta = current_time - self.last_processing_time
        throughput = 1 / time_delta if time_delta > 0 else 0
        
        # Get system metrics
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024**2  # MB
            cpu_usage = process.cpu_percent()
        except ImportError:
            memory_usage = 0
            cpu_usage = 0
        
        # Create metrics
        metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            current_regime=int(result.regime_predictions[-1]) if len(result.regime_predictions) > 0 else 0,
            regime_confidence=float(np.mean(result.regime_probabilities[-1])) if len(result.regime_probabilities) > 0 else 0.0,
            economic_score=float(np.mean(result.economic_significance_scores)) if len(result.economic_significance_scores) > 0 else 0.0,
            trading_score=float(np.mean(result.trading_viability_scores)) if len(result.trading_viability_scores) > 0 else 0.0,
            stability_score=float(np.mean(result.regime_stability_scores)) if len(result.regime_stability_scores) > 0 else 0.0,
            processing_latency_ms=processing_time * 1000,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            throughput_samples_per_second=throughput
        )
        
        self.metrics_history.append(metrics)
        self.processed_samples += 1
        self.last_processing_time = current_time
        
        tprint_debug(f"📊 Metrics recorded: regime={metrics.current_regime}, latency={metrics.processing_latency_ms:.1f}ms")
    
    def get_current_metrics(self) -> Optional[RealTimeMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        metrics = list(self.metrics_history)
        
        return {
            'total_samples_processed': self.processed_samples,
            'uptime_seconds': time.time() - self.start_time,
            'average_latency_ms': np.mean([m.processing_latency_ms for m in metrics]),
            'average_throughput_sps': np.mean([m.throughput_samples_per_second for m in metrics]),
            'average_memory_usage_mb': np.mean([m.memory_usage_mb for m in metrics]),
            'average_cpu_usage_percent': np.mean([m.cpu_usage_percent for m in metrics]),
            'current_regime': metrics[-1].current_regime if metrics else None,
            'regime_changes': len([m for i, m in enumerate(metrics[1:]) if m.current_regime != metrics[i].current_regime])
        }

class RealTimeRegimeMonitor:
    """Main real-time regime monitoring system."""
    
    def __init__(self, config: UnifiedRegimeConfig, 
                 data_source: Optional[Callable] = None,
                 event_callbacks: Optional[List[Callable]] = None):
        """Initialize real-time regime monitor."""
        tprint_info("🚀 Initializing Real-Time Regime Monitor")
        
        self.config = config
        self.data_source = data_source
        self.event_callbacks = event_callbacks or []
        
        # Initialize components
        self.regime_detector = UnifiedRegimeDetector(config)
        self.data_processor = DataStreamProcessor(
            window_size=config.min_regime_samples,
            overlap=config.min_regime_samples // 2
        )
        self.change_detector = RegimeChangeDetector()
        self.performance_monitor = PerformanceMonitor()
        self.performance_optimizer = get_performance_optimizer()
        
        # Real-time state
        self.is_running = False
        self.processing_thread = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=100)
        
        # Statistics
        self.total_events = 0
        self.last_regime = None
        
        tprint_success("✅ Real-Time Regime Monitor initialized")
    
    def add_event_callback(self, callback: Callable):
        """Add an event callback function."""
        self.event_callbacks.append(callback)
        tprint_debug(f"📞 Event callback added: {callback.__name__}")
    
    def start_monitoring(self, async_mode: bool = False):
        """Start real-time monitoring."""
        if self.is_running:
            tprint_warning("⚠️ Monitoring already running")
            return
        
        self.is_running = True
        tprint_info("🔄 Starting real-time regime monitoring")
        
        if async_mode:
            # Start async monitoring
            asyncio.create_task(self._async_monitoring_loop())
        else:
            # Start thread-based monitoring
            self.processing_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.processing_thread.start()
        
        tprint_success("✅ Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_running:
            tprint_warning("⚠️ Monitoring not running")
            return
        
        self.is_running = False
        tprint_info("⏹️ Stopping real-time regime monitoring")
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        tprint_success("✅ Real-time monitoring stopped")
    
    def add_market_data(self, data_point: Dict[str, float], timestamp: Optional[datetime] = None):
        """Add market data point for processing."""
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            self.data_queue.put_nowait((data_point, timestamp))
            tprint_debug(f"📈 Market data queued: {timestamp}")
        except queue.Full:
            tprint_warning("⚠️ Data queue full, dropping data point")
    
    def _monitoring_loop(self):
        """Main monitoring loop (thread-based)."""
        tprint_info("🔄 Monitoring loop started")
        
        while self.is_running:
            try:
                # Process queued data
                self._process_queued_data()
                
                # Check for regime changes
                self._check_regime_changes()
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                tprint_error(f"❌ Monitoring loop error: {e}")
                time.sleep(1.0)
        
        tprint_info("⏹️ Monitoring loop stopped")
    
    async def _async_monitoring_loop(self):
        """Main monitoring loop (async)."""
        tprint_info("🔄 Async monitoring loop started")
        
        while self.is_running:
            try:
                # Process queued data
                await self._async_process_queued_data()
                
                # Check for regime changes
                self._check_regime_changes()
                
                # Async sleep
                await asyncio.sleep(0.1)
                
            except Exception as e:
                tprint_error(f"❌ Async monitoring loop error: {e}")
                await asyncio.sleep(1.0)
        
        tprint_info("⏹️ Async monitoring loop stopped")
    
    def _process_queued_data(self):
        """Process queued market data."""
        processed_count = 0
        
        while not self.data_queue.empty() and processed_count < 10:  # Process up to 10 items per cycle
            try:
                data_point, timestamp = self.data_queue.get_nowait()
                
                # Add to data processor
                self.data_processor.add_data_point(data_point, timestamp)
                
                # Check if we should process
                if self.data_processor.should_process():
                    self._process_regime_detection()
                
                processed_count += 1
                
            except queue.Empty:
                break
            except Exception as e:
                tprint_error(f"❌ Data processing error: {e}")
    
    async def _async_process_queued_data(self):
        """Process queued market data (async)."""
        processed_count = 0
        
        while not self.data_queue.empty() and processed_count < 10:
            try:
                data_point, timestamp = self.data_queue.get_nowait()
                
                # Add to data processor
                self.data_processor.add_data_point(data_point, timestamp)
                
                # Check if we should process
                if self.data_processor.should_process():
                    await self._async_process_regime_detection()
                
                processed_count += 1
                
            except queue.Empty:
                break
            except Exception as e:
                tprint_error(f"❌ Async data processing error: {e}")
    
    def _process_regime_detection(self):
        """Process regime detection on current window."""
        start_time = time.time()
        
        try:
            # Get processing window
            window_data, window_timestamps = self.data_processor.get_processing_window()
            if window_data is None:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(window_data, columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Detect regimes
            result = self.regime_detector.detect_regimes(df, window_timestamps)
            
            if result.success:
                # Record performance metrics
                processing_time = time.time() - start_time
                self.performance_monitor.record_processing(result, processing_time)
                
                # Check for regime changes
                current_regime = int(result.regime_predictions[-1])
                confidence = float(np.mean(result.regime_probabilities[-1]))
                economic_sig = float(np.mean(result.economic_significance_scores))
                trading_viab = float(np.mean(result.trading_viability_scores))
                
                change_event = self.change_detector.detect_change(
                    current_regime, confidence, economic_sig, trading_viab,
                    datetime.now()
                )
                
                if change_event:
                    self._handle_regime_change(change_event)
                
                self.last_regime = current_regime
                
                tprint_debug(f"🔄 Regime detection: {current_regime} (confidence: {confidence:.3f})")
            
        except Exception as e:
            tprint_error(f"❌ Regime detection error: {e}")
    
    async def _async_process_regime_detection(self):
        """Process regime detection on current window (async)."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self._process_regime_detection)
    
    def _check_regime_changes(self):
        """Check for pending regime change events."""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._handle_regime_change(event)
            except queue.Empty:
                break
    
    def _handle_regime_change(self, event: RegimeChangeEvent):
        """Handle a regime change event."""
        self.total_events += 1
        
        tprint_warning(f"🔄 Regime change event #{self.total_events}: {event.from_regime} → {event.to_regime}")
        
        # Call event callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                tprint_error(f"❌ Event callback error: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        current_metrics = self.performance_monitor.get_current_metrics()
        performance_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'is_running': self.is_running,
            'current_regime': self.last_regime,
            'total_events': self.total_events,
            'data_queue_size': self.data_queue.qsize(),
            'event_queue_size': self.event_queue.qsize(),
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'performance_summary': performance_summary,
            'change_history_length': len(self.change_detector.change_history)
        }
    
    def get_regime_history(self, limit: int = 100) -> List[RealTimeMetrics]:
        """Get recent regime history."""
        return list(self.performance_monitor.metrics_history)[-limit:]
    
    def get_change_events(self, limit: int = 50) -> List[RegimeChangeEvent]:
        """Get recent regime change events."""
        return list(self.change_detector.change_history)[-limit:]

# Utility functions for real-time monitoring

def create_real_time_monitor(config: Optional[UnifiedRegimeConfig] = None,
                           data_source: Optional[Callable] = None) -> RealTimeRegimeMonitor:
    """Create a real-time regime monitor with default configuration."""
    if config is None:
        config = UnifiedRegimeConfig.create_production_config()
    
    return RealTimeRegimeMonitor(config, data_source)

def market_data_callback(event: RegimeChangeEvent):
    """Default market data callback for regime changes."""
    tprint(f"📊 Regime Change: {event.from_regime} → {event.to_regime} at {event.timestamp}", 
           color="yellow")
    tprint(f"   Confidence: {event.confidence:.3f}, Economic: {event.economic_significance:.3f}", 
           color="white")

# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Create real-time monitor
    config = UnifiedRegimeConfig.create_production_config()
    monitor = create_real_time_monitor(config)
    
    # Add event callback
    monitor.add_event_callback(market_data_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    tprint("🚀 Real-time monitoring started. Simulating market data...")
    
    try:
        # Simulate market data for 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            # Generate random market data
            price = 100 + random.uniform(-5, 5)
            volume = random.uniform(1000, 10000)
            
            data_point = {
                'open': price,
                'high': price + random.uniform(0, 2),
                'low': price - random.uniform(0, 2),
                'close': price + random.uniform(-1, 1),
                'volume': volume
            }
            
            monitor.add_market_data(data_point)
            
            # Print status every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                status = monitor.get_current_status()
                tprint(f"📊 Status: Regime {status['current_regime']}, Events: {status['total_events']}, "
                      f"Queue: {status['data_queue_size']}")
            
            time.sleep(1)  # 1 second intervals
    
    except KeyboardInterrupt:
        tprint("⏹️ Stopping monitoring...")
    
    finally:
        monitor.stop_monitoring()
        
        # Print final summary
        status = monitor.get_current_status()
        tprint("📊 FINAL SUMMARY:", color="cyan", bold=True)
        tprint(f"   Total Events: {status['total_events']}")
        tprint(f"   Final Regime: {status['current_regime']}")
        tprint(f"   Performance: {status['performance_summary']}")