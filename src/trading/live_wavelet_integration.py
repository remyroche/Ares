"""
Live Trading Wavelet Integration

This module integrates the computationally-aware wavelet analyzer
into the live trading pipeline with performance monitoring.
"""

from src.utils.logger import system_logger
from typing import Any
import time

from src.trading.live_wavelet_analyzer import LiveWaveletAnalyzer, WaveletSignal
from src.utils.error_handler import handle_errors
import numpy as np
import pandas as pd

class LiveWaveletIntegration:
    """
    Integration layer for wavelet analysis in live trading.

    Provides:
    - Performance monitoring
    - Signal validation
    - Integration with existing trading pipeline
    - Fallback mechanisms
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LiveWaveletIntegration")

        # Wavelet analyzer
        self.wavelet_analyzer: LiveWaveletAnalyzer | None = None

        # Performance monitoring
        self.performance_stats = {}
        self.signal_history = []
        self.is_enabled = config.get("enable_live_wavelet", True)

        # Integration settings
        self.signal_weight = config.get("wavelet_signal_weight", 0.3)
        self.min_confidence = config.get("min_wavelet_confidence", 0.6)
        self.max_signal_age = config.get("max_signal_age", 60)  # seconds

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="live wavelet integration initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="wavelet signal processing",
    )
    async def process_market_data(
        self, market_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Process market data and generate wavelet signals.

        Args:
            market_data: Market data from trading pipeline

        Returns:
            Wavelet analysis results or None
        """
        try:
            if not self.is_enabled or not self.wavelet_analyzer:
                return None

            # Extract price and volume data
            price_data = self._extract_price_data(market_data)
            volume_data = self._extract_volume_data(market_data)

            if price_data is None or price_data.empty:
                return None

            # Generate wavelet signal
            signal = await self.wavelet_analyzer.generate_signal(
                price_data, volume_data,
            )

            if signal is None:
                return None

            # Correlation context for observability
            correlation_id = market_data.get("correlation_id") or market_data.get(
                "order_link_id",
            )
            log = (
                self.logger
                if correlation_id is None
                else self.logger.getChild(str(correlation_id))
            )

            # Validate signal
            if not self._validate_signal(signal):
                log.info("Wavelet signal rejected by validator")
                return None

            # Create analysis results
            results = self._create_analysis_results(signal, market_data)

            # Update performance stats
            self._update_performance_stats(signal)

            log.info("Wavelet signal processed")
            return results

        except (ValueError, KeyError) as e:
            self.logger.error(f"Error processing market data - Invalid data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing market data - Unexpected error: {e}")
            return None

    def _extract_price_data(
        self, market_data: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Extract price data from market data."""
        try:
            if "price_data" in market_data:
                return market_data["price_data"]
            if "ohlcv" in market_data:
                return pd.DataFrame(market_data["ohlcv"])
            if "close" in market_data:
                # Single price point
                return pd.DataFrame(
                    {
                        "close": [market_data["close"]],
                        "open": [market_data.get("open", market_data["close"])],
                        "high": [market_data.get("high", market_data["close"])],
                        "low": [market_data.get("low", market_data["close"])],
                        "volume": [market_data.get("volume", 0)],
                    },
                )
            return None

        except (KeyError, ValueError) as e:
            self.logger.error(
                f"Error extracting price data - Invalid data structure: {e}",
            )
            return None
        except Exception as e:
            self.logger.error(f"Error extracting price data - Unexpected error: {e}")
            return None

    def _extract_volume_data(
        self, market_data: dict[str, Any],
    ) -> pd.DataFrame | None:
        """Extract volume data from market data."""
        try:
            if "volume_data" in market_data:
                return market_data["volume_data"]
            if "volume" in market_data:
                return pd.DataFrame({"volume": [market_data["volume"]]})
            return None

        except (KeyError, ValueError) as e:
            self.logger.error(
                f"Error extracting volume data - Invalid data structure: {e}",
            )
            return None
        except Exception as e:
            self.logger.error(f"Error extracting volume data - Unexpected error: {e}")
            return None

    def _validate_signal(self, signal: WaveletSignal) -> bool:
        """Validate wavelet signal."""
        try:
            # Check confidence threshold
            if signal.confidence < self.min_confidence:
                return False

            # Check signal age
            signal_age = time.time() - signal.timestamp
            if signal_age > self.max_signal_age:
                return False

            # Check computation time
            if signal.computation_time > 0.1:  # 100ms threshold
                return False

            return True

        except (AttributeError, TypeError) as e:
            self.logger.error(f"Error validating signal - Invalid signal object: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating signal - Unexpected error: {e}")
            return False

    def _create_analysis_results(
        self, signal: WaveletSignal,
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Create analysis results for trading pipeline."""
        try:
            return {
                "wavelet_signal": signal.signal_type,
                "wavelet_confidence": signal.confidence,
                "wavelet_energy": signal.energy_level,
                "wavelet_entropy": signal.entropy_level,
                "wavelet_computation_time": signal.computation_time,
                "wavelet_timestamp": signal.timestamp,
                # Integration with existing pipeline
                "technical_analysis": {
                    "wavelet_trend": signal.signal_type,
                    "wavelet_strength": signal.confidence,
                    "wavelet_volatility": signal.entropy_level,
                    "wavelet_momentum": signal.energy_level,
                },
                # Signal generation
                "signal_generation": {
                    "wavelet_signal": signal.signal_type,
                    "wavelet_confidence": signal.confidence,
                    "combined_signal": self._combine_with_existing_signals(
                        signal, market_data,
                    ),
                },
                # Performance metrics
                "performance": {
                    "wavelet_computation_time": signal.computation_time,
                    "signal_age": time.time() - signal.timestamp,
                    "signal_quality": signal.confidence,
                },
            }

        except (AttributeError, KeyError) as e:
            self.logger.error(
                f"Error creating analysis results - Invalid signal data: {e}",
            )
            return {}
        except Exception as e:
            self.logger.error(
                f"Error creating analysis results - Unexpected error: {e}",
            )
            return {}

    def _combine_with_existing_signals(
        self, wavelet_signal: WaveletSignal,
        market_data: dict[str, Any],
    ) -> str:
        """Combine wavelet signal with existing trading signals."""
        try:
            # Get existing signals from market data
            existing_signals = market_data.get("signal_generation", {})

            # Simple combination logic
            if wavelet_signal.signal_type == "hold":
                return "hold"

            # If wavelet signal is strong, use it
            if wavelet_signal.confidence > 0.8:
                return wavelet_signal.signal_type

            # Otherwise, combine with existing signals
            existing_signal = existing_signals.get("primary_signal", "hold")

            if existing_signal == wavelet_signal.signal_type:
                return wavelet_signal.signal_type  # Agreement
            if existing_signal == "hold":
                return wavelet_signal.signal_type  # Wavelet provides signal
            return "hold"  # Disagreement = be conservative

        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return "hold"

    def _update_performance_stats(self, signal: WaveletSignal) -> None:
        """Update performance statistics."""
        try:
            # Update signal history
            self.signal_history.append(
                {
                    "timestamp": signal.timestamp,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "computation_time": signal.computation_time,
                },
            )

            # Keep only recent history
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]

            # Update performance stats
            if self.wavelet_analyzer:
                self.performance_stats = self.wavelet_analyzer.get_performance_stats()

        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")

    def get_latest_signal(self) -> WaveletSignal | None:
        """Get the latest wavelet signal."""
        if self.wavelet_analyzer:
            return self.wavelet_analyzer.get_latest_signal()
        return None

    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()
        if self.wavelet_analyzer:
            self.wavelet_analyzer.clear_history()
        self.logger.info("Wavelet signal history cleared")
