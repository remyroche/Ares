"""
Live Trading Wavelet Analyzer - Computationally Aware Implementation

This module provides a lightweight = real-time wavelet analysis system
optimized for live trading with strict performance constraints.
"""

from collections import deque
from src.utils.logger import system_logger
from typing import Any
import asyncio
import threading
import time

from dataclasses import dataclass
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, initialization_error, timeout, warning
import numpy as np
import pandas as pd
import pywt

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
        self.max_data_points = config.get(
            "max_data_points",
            256,
        )  # Power of 2 for efficiency
        self.sliding_window_size = config.get("sliding_window_size", 128)

        # Wavelet configuration (minimal for speed)
        self.wavelet_type = config.get("wavelet_type", "db4")  # Single type
        self.decomposition_level = config.get(
            "decomposition_level",
            2,
        )  # Minimal levels
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
        self.latest_signal: WaveletSignal | None = None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="live wavelet analyzer initialization",
    )
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.max_computation_time > 0.5:
            self.print(warning("Max computation time too high for live trading"))
            self.max_computation_time = 0.1

        if self.sliding_window_size > 512:
            self.print(warning("Sliding window too large for live trading"))
            self.sliding_window_size = 256

        # Ensure window size is power of 2 for efficient wavelet computation
        if self.sliding_window_size & self.sliding_window_size - 1 != 0:
            self.sliding_window_size = 2 ** (self.sliding_window_size - 1).bit_length()
            self.logger.info(f"Adjusted window size to {self.sliding_window_size}")

    def _precompute_wavelet_coeffs(self) -> None:
        """Pre-compute wavelet coefficients for efficiency."""
        try:
            # Create a dummy signal for coefficient computation
            dummy_signal = np.random.randn(self.sliding_window_size).astype(
                np.float32, copy=False,
            )

            # Pre-compute DWT coefficients structure
            self.wavelet_obj = pywt.Wavelet(self.wavelet_type)
            level = self._get_decomposition_level(len(dummy_signal))
            self.dwt_coeffs_structure = pywt.wavedec(
                dummy_signal, self.wavelet_obj,
                level=level, mode=self.padding_mode,
            )

            self.logger.info("✅ Pre-computed wavelet coefficients")

        except Exception as e:
            self.logger.error(f"Error pre-computing wavelet coefficients: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="live wavelet signal generation",
    )
    def _update_sliding_windows(
        self, price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None
    ) -> None:
        """Update sliding windows with new data."""
        try:
            # Extract latest price differences (stationary series)
            if len(price_data) > 0:
                latest_close = price_data["close"].iloc[-1]
                if len(self.price_window) > 0:
                    price_diff = latest_close - self.price_window[-1]
                else:
                    price_diff = 0.0

                self.price_window.append(latest_close)

            # Update volume window if available
            if volume_data is not None and len(volume_data) > 0:
                latest_volume = volume_data["volume"].iloc[-1]
                self.volume_window.append(latest_volume)

        except Exception as e:
            self.logger.error(f"Error updating sliding windows: {e}")

    async def _perform_fast_wavelet_analysis(self) -> WaveletSignal | None:
        """Perform fast wavelet analysis with timeout."""
        try:
            # Convert price window to numpy array
            price_array = np.array(list(self.price_window))

            # Use asyncio to enforce timeout
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._compute_wavelet_features,
                price_array
            )

            if result is None:
                return None

            # Generate trading signal
            return self._generate_trading_signal(result)

        except TimeoutError:
            self.logger.error("Wavelet computation timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error in fast wavelet analysis: {e}")
            return None

    def _generate_trading_signal(self, features: dict[str, float]) -> WaveletSignal:
        """Generate trading signal from wavelet features."""
        try:
            # Extract key metrics
            energy_features = {k: v for k, v in features.items() if "energy" in k}
            entropy_features = {k: v for k, v in features.items() if "entropy" in k}

            # Calculate average energy and entropy
            avg_energy = np.mean(
                list(energy_features.values()) if energy_features else [0.0],
            )
            avg_entropy = np.mean(
                list(entropy_features.values()) if entropy_features else [0.0],
            )

            # Simple signal generation logic
            signal_type = "hold"
            confidence = 0.5

            # High energy + low entropy = strong trend (buy)
            if (
                avg_energy > self.energy_threshold
                and avg_entropy < self.entropy_threshold
            ):
                signal_type = "buy"
                confidence = min(0.9, avg_energy / self.energy_threshold)

            # Low energy + high entropy = reversal (sell)
            elif (
                avg_energy < self.energy_threshold * 0.5
                and avg_entropy > self.entropy_threshold
            ):
                signal_type = "sell"
                confidence = min(0.9, avg_entropy / self.entropy_threshold)

            # Create signal
            return WaveletSignal(
                timestamp=time.time(),
                signal_type=signal_type,
                confidence=confidence,
                energy_level=avg_energy,
                entropy_level=avg_entropy,
                computation_time=0.0,  # Will be set by caller
            )

        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return WaveletSignal(
                timestamp=time.time(),
                signal_type="hold",
                confidence=0.0,
                energy_level=0.0,
                entropy_level=0.0,
                computation_time=0.0,
            )

    def get_latest_signal(self) -> WaveletSignal | None:
        """Get the latest wavelet signal."""
        return self.latest_signal
