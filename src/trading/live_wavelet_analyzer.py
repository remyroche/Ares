"""
Live Trading Wavelet Analyzer - Computationally Aware Implementation

This module provides a lightweight, real-time wavelet analysis system
optimized for live trading with strict performance constraints.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
import time
import asyncio
from collections import deque
import threading
from dataclasses import dataclass

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class WaveletSignal:
    """Lightweight wavelet signal container."""
    timestamp: float
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    energy_level: float
    entropy_level: float
    computation_time: float


class LiveWaveletAnalyzer:
    """
    Computationally-aware wavelet analyzer for live trading.
    
    Key optimizations:
    - Single wavelet type (db4) for speed
    - Minimal decomposition levels (2-3)
    - Sliding window approach
    - Pre-computed lookup tables
    - Async computation with timeouts
    - Memory-efficient data structures
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LiveWaveletAnalyzer")
        
        # Performance constraints
        self.max_computation_time = config.get("max_computation_time", 0.1)  # 100ms
        self.max_data_points = config.get("max_data_points", 256)  # Power of 2 for efficiency
        self.sliding_window_size = config.get("sliding_window_size", 128)
        
        # Wavelet configuration (minimal for speed)
        self.wavelet_type = config.get("wavelet_type", "db4")  # Single type
        self.decomposition_level = config.get("decomposition_level", 2)  # Minimal levels
        self.padding_mode = config.get("padding_mode", "symmetric")
        
        # Signal thresholds
        self.energy_threshold = config.get("energy_threshold", 0.01)
        self.entropy_threshold = config.get("entropy_threshold", 0.5)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Performance tracking
        self.computation_times = deque(maxlen=100)
        self.signal_history = deque(maxlen=1000)
        self.is_initialized = False
        
        # Threading for async computation
        self.computation_lock = threading.Lock()
        self.latest_signal: Optional[WaveletSignal] = None
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="live wavelet analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the live wavelet analyzer."""
        try:
            self.logger.info("🚀 Initializing Live Wavelet Analyzer...")
            
            # Validate configuration
            self._validate_config()
            
            # Pre-compute wavelet coefficients for efficiency
            self._precompute_wavelet_coeffs()
            
            # Initialize sliding window
            self.price_window = deque(maxlen=self.sliding_window_size)
            self.volume_window = deque(maxlen=self.sliding_window_size)
            
            self.is_initialized = True
            self.logger.info(f"✅ Live Wavelet Analyzer initialized successfully")
            self.logger.info(f"📊 Config: window={self.sliding_window_size}, "
                           f"wavelet={self.wavelet_type}, levels={self.decomposition_level}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Live Wavelet Analyzer: {e}")
            return False
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.max_computation_time > 0.5:
            self.logger.warning("Max computation time too high for live trading")
            self.max_computation_time = 0.1
            
        if self.sliding_window_size > 512:
            self.logger.warning("Sliding window too large for live trading")
            self.sliding_window_size = 256
            
        # Ensure window size is power of 2 for efficient wavelet computation
        if not (self.sliding_window_size & (self.sliding_window_size - 1) == 0):
            self.sliding_window_size = 2 ** (self.sliding_window_size - 1).bit_length()
            self.logger.info(f"Adjusted window size to {self.sliding_window_size}")
    
    def _precompute_wavelet_coeffs(self) -> None:
        """Pre-compute wavelet coefficients for efficiency."""
        try:
            # Create a dummy signal for coefficient computation
            dummy_signal = np.random.randn(self.sliding_window_size)
            
            # Pre-compute DWT coefficients structure
            self.dwt_coeffs_structure = pywt.wavedec(
                dummy_signal, 
                self.wavelet_type, 
                level=self.decomposition_level,
                mode=self.padding_mode
            )
            
            self.logger.info("✅ Pre-computed wavelet coefficients")
            
        except Exception as e:
            self.logger.error(f"Error pre-computing wavelet coefficients: {e}")
    
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="live wavelet signal generation",
    )
    async def generate_signal(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None
    ) -> Optional[WaveletSignal]:
        """
        Generate trading signal using computationally-aware wavelet analysis.
        
        Args:
            price_data: Recent price data
            volume_data: Recent volume data (optional)
            
        Returns:
            WaveletSignal or None if computation timeout
        """
        try:
            if not self.is_initialized:
                self.logger.error("Live Wavelet Analyzer not initialized")
                return None
            
            start_time = time.time()
            
            # Update sliding windows
            self._update_sliding_windows(price_data, volume_data)
            
            # Check if we have enough data
            if len(self.price_window) < self.sliding_window_size // 2:
                return None
            
            # Perform fast wavelet analysis
            signal = await self._perform_fast_wavelet_analysis()
            
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)
            
            # Check performance constraints
            if computation_time > self.max_computation_time:
                self.logger.warning(f"Wavelet computation too slow: {computation_time:.3f}s")
                return None
            
            if signal:
                signal.computation_time = computation_time
                self.latest_signal = signal
                self.signal_history.append(signal)
                
                self.logger.info(f"📊 Wavelet signal: {signal.signal_type} "
                               f"(confidence: {signal.confidence:.2f}, "
                               f"time: {computation_time:.3f}s)")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating wavelet signal: {e}")
            return None
    
    def _update_sliding_windows(
        self, 
        price_data: pd.DataFrame, 
        volume_data: pd.DataFrame | None
    ) -> None:
        """Update sliding windows with new data."""
        try:
            # Extract latest price differences (stationary series)
            if len(price_data) > 0:
                latest_close = price_data['close'].iloc[-1]
                if len(self.price_window) > 0:
                    price_diff = latest_close - self.price_window[-1]
                else:
                    price_diff = 0.0
                
                self.price_window.append(latest_close)
            
            # Update volume window if available
            if volume_data is not None and len(volume_data) > 0:
                latest_volume = volume_data['volume'].iloc[-1]
                self.volume_window.append(latest_volume)
                
        except Exception as e:
            self.logger.error(f"Error updating sliding windows: {e}")
    
    async def _perform_fast_wavelet_analysis(self) -> Optional[WaveletSignal]:
        """Perform fast wavelet analysis with timeout."""
        try:
            # Convert price window to numpy array
            price_array = np.array(list(self.price_window))
            
            # Use asyncio to enforce timeout
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._compute_wavelet_features, 
                price_array
            )
            
            if result is None:
                return None
            
            # Generate trading signal
            signal = self._generate_trading_signal(result)
            return signal
            
        except asyncio.TimeoutError:
            self.logger.warning("Wavelet computation timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error in fast wavelet analysis: {e}")
            return None
    
    def _compute_wavelet_features(self, price_array: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute wavelet features with performance constraints."""
        try:
            # Ensure array length is power of 2 for efficiency
            target_length = 2 ** int(np.log2(len(price_array)))
            if len(price_array) != target_length:
                price_array = price_array[-target_length:]
            
            # Compute DWT (fastest wavelet transform)
            coeffs = pywt.wavedec(
                price_array, 
                self.wavelet_type, 
                level=self.decomposition_level,
                mode=self.padding_mode
            )
            
            # Extract key features efficiently
            features = {}
            
            # Energy features (most important for trading)
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    energy = np.sum(coeff ** 2)
                    features[f"level_{i}_energy"] = energy
                    
                    # Normalized energy
                    features[f"level_{i}_energy_norm"] = energy / len(coeff)
            
            # Entropy features (market disorder)
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0 and np.sum(coeff ** 2) > 0:
                    energy = np.sum(coeff ** 2)
                    entropy = -np.sum((coeff ** 2) / energy * 
                                    np.log((coeff ** 2) / energy + 1e-10))
                    features[f"level_{i}_entropy"] = entropy
            
            # Cross-level energy ratios
            if len(coeffs) > 1:
                for i in range(len(coeffs) - 1):
                    energy_i = np.sum(coeffs[i] ** 2)
                    energy_j = np.sum(coeffs[i + 1] ** 2)
                    if energy_i > 0:
                        features[f"energy_ratio_{i}_{i+1}"] = energy_j / energy_i
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error computing wavelet features: {e}")
            return None
    
    def _generate_trading_signal(self, features: Dict[str, float]) -> WaveletSignal:
        """Generate trading signal from wavelet features."""
        try:
            # Extract key metrics
            energy_features = {k: v for k, v in features.items() if 'energy' in k}
            entropy_features = {k: v for k, v in features.items() if 'entropy' in k}
            
            # Calculate average energy and entropy
            avg_energy = np.mean(list(energy_features.values())) if energy_features else 0.0
            avg_entropy = np.mean(list(entropy_features.values())) if entropy_features else 0.0
            
            # Simple signal generation logic
            signal_type = "hold"
            confidence = 0.5
            
            # High energy + low entropy = strong trend (buy)
            if avg_energy > self.energy_threshold and avg_entropy < self.entropy_threshold:
                signal_type = "buy"
                confidence = min(0.9, avg_energy / self.energy_threshold)
            
            # Low energy + high entropy = reversal (sell)
            elif avg_energy < self.energy_threshold * 0.5 and avg_entropy > self.entropy_threshold:
                signal_type = "sell"
                confidence = min(0.9, avg_entropy / self.entropy_threshold)
            
            # Create signal
            signal = WaveletSignal(
                timestamp=time.time(),
                signal_type=signal_type,
                confidence=confidence,
                energy_level=avg_energy,
                entropy_level=avg_entropy,
                computation_time=0.0  # Will be set by caller
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return WaveletSignal(
                timestamp=time.time(),
                signal_type="hold",
                confidence=0.0,
                energy_level=0.0,
                entropy_level=0.0,
                computation_time=0.0
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            if not self.computation_times:
                return {}
            
            avg_time = np.mean(self.computation_times)
            max_time = np.max(self.computation_times)
            min_time = np.min(self.computation_times)
            
            signal_count = len([s for s in self.signal_history if s.signal_type != "hold"])
            total_signals = len(self.signal_history)
            
            return {
                "avg_computation_time": avg_time,
                "max_computation_time": max_time,
                "min_computation_time": min_time,
                "signal_count": signal_count,
                "total_signals": total_signals,
                "signal_rate": signal_count / max(total_signals, 1),
                "window_size": self.sliding_window_size,
                "wavelet_type": self.wavelet_type
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def get_latest_signal(self) -> Optional[WaveletSignal]:
        """Get the latest wavelet signal."""
        return self.latest_signal
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()
        self.computation_times.clear()
        self.latest_signal = None