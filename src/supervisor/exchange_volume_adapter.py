#!/usr/bin/env python3
"""
Exchange Volume Adapter for Model Transfer Learning

This module handles the adaptation of models trained on high-volume exchanges
(Binance) to work effectively on lower-volume exchanges (MEXC = Gate.io).
"""

from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import (
    error,
    execution_error,
    initialization_error,
    invalid,
    warning
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
            True
        )
        self.enable_dynamic_adjustment: bool = self.adapter_config.get(
            "enable_dynamic_adjustment",
            True
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

        except Exception as e:
            self.logger.error(f"Error loading adapter configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate adapter configuration."""
        try:
            if self.volume_history_window <= 0:
                self.logger.error("Invalid volume history window")
                return False

            if not self.volume_profiles:
                self.logger.error("No volume profiles defined")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="volume metrics initialization",
    )
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
            Tuple of (should_execute = reason)
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
                return (False, f"Volume too low: {current_volume} < {min_volume}")

            return (True, "Trade execution approved")

        except Exception as e:
            self.print(execution_error(f"Error checking trade execution: {e}"))
            return (False, f"Error: {e}")

    @handle_errors(
        exceptions=(KeyError, ValueError),
        default_return=1.0,
        context="adaptation factor retrieval",
    )
@handle_errors(
    exceptions=(Exception,),
    default_return=None, context="exchange volume adapter setup",
)