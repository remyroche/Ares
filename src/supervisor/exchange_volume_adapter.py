#!/usr/bin/env python3
"""
Exchange Volume Adapter for Model Transfer Learning

This module handles the adaptation of models trained on high-volume exchanges
(Binance) to work effectively on lower-volume exchanges (MEXC, Gate.io).
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    warning,
)


class ExchangeVolumeAdapter:
    """
    Adapts trading strategies and position sizing based on exchange volume characteristics.

    This class handles the critical differences between exchanges:
    - Volume/liquidity differences
    - Spread and slippage variations
    - Market impact considerations
    - Data quality adjustments
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ExchangeVolumeAdapter")

        # Exchange volume profiles
        self.volume_profiles: dict[str, dict[str, Any]] = {
            "BINANCE": {
                "avg_daily_volume": 1000000,  # High volume
                "spread_multiplier": 1.0,  # Baseline
                "slippage_multiplier": 1.0,  # Baseline
                "position_size_multiplier": 1.0,  # Baseline
                "data_quality_score": 0.95,  # High quality
                "market_impact_threshold": 0.001,  # Low impact
            },
            "MEXC": {
                "avg_daily_volume": 50000,  # Lower volume
                "spread_multiplier": 2.5,  # Wider spreads
                "slippage_multiplier": 3.0,  # Higher slippage
                "position_size_multiplier": 0.4,  # Smaller positions
                "data_quality_score": 0.75,  # Moderate quality
                "market_impact_threshold": 0.005,  # Higher impact
            },
            "GATEIO": {
                "avg_daily_volume": 30000,  # Lower volume
                "spread_multiplier": 3.0,  # Wider spreads
                "slippage_multiplier": 3.5,  # Higher slippage
                "position_size_multiplier": 0.3,  # Smaller positions
                "data_quality_score": 0.70,  # Moderate quality
                "market_impact_threshold": 0.008,  # Higher impact
            },
        }

        # Configuration
        self.adapter_config: dict[str, Any] = self.config.get(
            "exchange_volume_adapter",
            {},
        )
        self.enable_volume_adaptation: bool = self.adapter_config.get(
            "enable_volume_adaptation",
            True,
        )
        self.enable_dynamic_adjustment: bool = self.adapter_config.get(
            "enable_dynamic_adjustment",
            True,
        )
        self.volume_history_window: int = self.adapter_config.get(
            "volume_history_window",
            24,
        )  # hours

        # State
        self.current_volume_metrics: dict[str, dict[str, Any]] = {}
        self.adaptation_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid exchange volume adapter configuration"),
            AttributeError: (False, "Missing required adapter parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="exchange volume adapter initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the exchange volume adapter."""
        try:
            self.logger.info("Initializing Exchange Volume Adapter...")

            # Load configuration
            await self._load_adapter_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for exchange volume adapter"))
                return False

            # Initialize volume metrics
            await self._initialize_volume_metrics()

            self.logger.info(
                "✅ Exchange Volume Adapter initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Exchange Volume Adapter initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="adapter configuration loading",
    )
    async def _load_adapter_configuration(self) -> None:
        """Load adapter configuration."""
        try:
            # Set defaults
            self.adapter_config.setdefault("enable_volume_adaptation", True)
            self.adapter_config.setdefault("enable_dynamic_adjustment", True)
            self.adapter_config.setdefault("volume_history_window", 24)
            self.adapter_config.setdefault("min_volume_threshold", 1000)
            self.adapter_config.setdefault("max_position_size_reduction", 0.8)

            self.logger.info("Adapter configuration loaded successfully")

        except Exception:
            self.print(error("Error loading adapter configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate adapter configuration."""
        try:
            if self.volume_history_window <= 0:
                self.print(invalid("Invalid volume history window"))
                return False

            if not self.volume_profiles:
                self.print(error("No volume profiles defined"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="volume metrics initialization",
    )
    async def _initialize_volume_metrics(self) -> None:
        """Initialize volume metrics for all exchanges."""
        try:
            for exchange in self.volume_profiles:
                self.current_volume_metrics[exchange] = {
                    "current_volume": 0,
                    "avg_volume_24h": 0,
                    "volume_trend": 0,
                    "spread_adjustment": 1.0,
                    "slippage_adjustment": 1.0,
                    "last_updated": datetime.now(),
                }

            self.logger.info("Volume metrics initialized successfully")

        except Exception:
            self.print(initialization_error("Error initializing volume metrics: {e}"))

    def get_volume_profile(self, exchange: str) -> dict[str, Any]:
        """Get volume profile for a specific exchange."""
        exchange_upper = exchange.upper()
        if exchange_upper not in self.volume_profiles:
            self.logger.warning(
                f"No volume profile found for {exchange}, using BINANCE profile",
            )
            return self.volume_profiles["BINANCE"]
        return self.volume_profiles[exchange_upper]

    def calculate_position_size_adjustment(
        self,
        exchange: str,
        base_position_size: float,
        current_volume: float = None,
        confidence_score: float = None,
    ) -> float:
        """
        Calculate position size adjustment based on exchange volume characteristics.

        Args:
            exchange: Exchange name
            base_position_size: Base position size from model
            current_volume: Current market volume (optional)
            confidence_score: Model confidence score (optional)

        Returns:
            Adjusted position size
        """
        try:
            profile = self.get_volume_profile(exchange)
            base_multiplier = profile["position_size_multiplier"]

            # Start with base multiplier
            adjustment = base_multiplier

            # Adjust based on current volume if available
            if current_volume is not None:
                avg_volume = profile["avg_daily_volume"]
                volume_ratio = current_volume / avg_volume

                # Reduce position size if volume is low
                if volume_ratio < 0.5:
                    adjustment *= 0.7
                elif volume_ratio < 0.8:
                    adjustment *= 0.85
                elif volume_ratio > 1.5:
                    adjustment *= 1.1  # Slightly increase for high volume

            # Adjust based on confidence score
            if confidence_score is not None:
                if confidence_score < 0.6:
                    adjustment *= 0.5  # Reduce size for low confidence
                elif confidence_score > 0.9:
                    adjustment *= 1.2  # Increase size for high confidence

            # Apply maximum reduction limit
            max_reduction = self.adapter_config.get("max_position_size_reduction", 0.8)
            adjustment = max(adjustment, max_reduction)

            adjusted_size = base_position_size * adjustment

            self.logger.info(
                f"Position size adjustment for {exchange}: "
                f"base={base_position_size:.4f}, "
                f"adjustment={adjustment:.3f}, "
                f"final={adjusted_size:.4f}",
            )

            return adjusted_size

        except Exception:
            self.print(error("Error calculating position size adjustment: {e}"))
            return base_position_size * 0.5  # Conservative fallback

    def calculate_spread_adjustment(self, exchange: str, base_spread: float) -> float:
        """Calculate spread adjustment based on exchange characteristics."""
        try:
            profile = self.get_volume_profile(exchange)
            spread_multiplier = profile["spread_multiplier"]
            return base_spread * spread_multiplier

        except Exception:
            self.print(error("Error calculating spread adjustment: {e}"))
            return base_spread * 2.0  # Conservative fallback

    def calculate_slippage_adjustment(
        self,
        exchange: str,
        base_slippage: float,
    ) -> float:
        """Calculate slippage adjustment based on exchange characteristics."""
        try:
            profile = self.get_volume_profile(exchange)
            slippage_multiplier = profile["slippage_multiplier"]
            return base_slippage * slippage_multiplier

        except Exception:
            self.print(error("Error calculating slippage adjustment: {e}"))
            return base_slippage * 2.5  # Conservative fallback

    def adjust_model_confidence(
        self,
        exchange: str,
        base_confidence: float,
        data_quality_metrics: dict[str, Any] = None,
    ) -> float:
        """
        Adjust model confidence based on exchange data quality.

        Args:
            exchange: Exchange name
            base_confidence: Base confidence from model
            data_quality_metrics: Optional data quality metrics

        Returns:
            Adjusted confidence score
        """
        try:
            profile = self.get_volume_profile(exchange)
            data_quality_score = profile["data_quality_score"]

            # Adjust confidence based on data quality
            adjusted_confidence = base_confidence * data_quality_score

            # Further adjust based on data quality metrics if available
            if data_quality_metrics:
                # Example: adjust for missing data, noise, etc.
                if data_quality_metrics.get("missing_data_ratio", 0) > 0.1:
                    adjusted_confidence *= 0.8
                if data_quality_metrics.get("noise_level", 0) > 0.5:
                    adjusted_confidence *= 0.9

            # Ensure confidence stays within bounds
            adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))

            self.logger.info(
                f"Confidence adjustment for {exchange}: "
                f"base={base_confidence:.3f}, "
                f"adjusted={adjusted_confidence:.3f}",
            )

            return adjusted_confidence

        except Exception:
            self.print(error("Error adjusting model confidence: {e}"))
            return base_confidence * 0.8  # Conservative fallback

    def should_execute_trade(
        self,
        exchange: str,
        position_size: float,
        current_volume: float = None,
        market_impact_threshold: float = None,
    ) -> tuple[bool, str]:
        """
        Determine if a trade should be executed based on volume constraints.

        Args:
            exchange: Exchange name
            position_size: Position size in base currency
            current_volume: Current market volume
            market_impact_threshold: Custom market impact threshold

        Returns:
            Tuple of (should_execute, reason)
        """
        try:
            profile = self.get_volume_profile(exchange)
            threshold = market_impact_threshold or profile["market_impact_threshold"]

            if current_volume is None:
                current_volume = profile["avg_daily_volume"]

            # Calculate potential market impact
            impact_ratio = position_size / current_volume

            if impact_ratio > threshold:
                return (
                    False,
                    f"Market impact too high: {impact_ratio:.4f} > {threshold:.4f}",
                )

            # Check minimum volume threshold
            min_volume = self.adapter_config.get("min_volume_threshold", 1000)
            if current_volume < min_volume:
                return False, f"Volume too low: {current_volume} < {min_volume}"

            return True, "Trade execution approved"

        except Exception as e:
            self.print(execution_error("Error checking trade execution: {e}"))
            return False, f"Error: {e}"

    def get_adaptation_summary(self) -> dict[str, Any]:
        """Get summary of current volume adaptations."""
        try:
            return {
                "enabled": self.enable_volume_adaptation,
                "dynamic_adjustment": self.enable_dynamic_adjustment,
                "volume_profiles": self.volume_profiles,
                "current_metrics": self.current_volume_metrics,
                "adaptation_history_count": len(self.adaptation_history),
            }

        except Exception as e:
            self.print(error("Error getting adaptation summary: {e}"))
            return {"error": str(e)}

    async def update_volume_metrics(
        self,
        exchange: str,
        current_volume: float,
        spread: float = None,
        slippage: float = None,
    ) -> None:
        """Update volume metrics for an exchange."""
        try:
            if exchange.upper() not in self.current_volume_metrics:
                self.print(warning("No metrics tracking for {exchange}"))
                return

            metrics = self.current_volume_metrics[exchange.upper()]
            metrics["current_volume"] = current_volume
            metrics["last_updated"] = datetime.now()

            if spread is not None:
                metrics["spread_adjustment"] = spread
            if slippage is not None:
                metrics["slippage_adjustment"] = slippage

            # Store in history
            self.adaptation_history.append(
                {
                    "exchange": exchange,
                    "timestamp": datetime.now(),
                    "volume": current_volume,
                    "spread": spread,
                    "slippage": slippage,
                },
            )

            # Keep history within limits
            max_history = self.adapter_config.get("max_history", 1000)
            if len(self.adaptation_history) > max_history:
                self.adaptation_history = self.adaptation_history[-max_history:]

        except Exception:
            self.print(error("Error updating volume metrics: {e}"))

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("Cleaning up Exchange Volume Adapter...")
            # Clear history
            self.adaptation_history.clear()
            self.current_volume_metrics.clear()
            self.logger.info("✅ Exchange Volume Adapter cleanup completed")

        except Exception:
            self.print(error("Error during cleanup: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="exchange volume adapter setup",
)
async def setup_exchange_volume_adapter(
    config: dict[str, Any] | None = None,
) -> ExchangeVolumeAdapter | None:
    """Setup exchange volume adapter."""
    try:
        if config is None:
            config = {}

        adapter = ExchangeVolumeAdapter(config)
        if await adapter.initialize():
            return adapter
        return None

    except Exception:
        system_logger.exception(error("Error setting up exchange volume adapter: {e}"))
        return None
