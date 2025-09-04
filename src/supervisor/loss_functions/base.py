"""
Base class for PnL Loss Functions.

This module provides the base functionality and structure for all
PnL-related loss function calculators.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from copy import copy
import asyncio


class PnLLossFunctionsBase:
    """
    Base class for PnL Loss Functions with common functionality.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize base PnL loss functions.

        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild(self.__class__.__name__)

        # Common state
        self.is_calculating: bool = False
        self.calculation_results: Dict[str, Any] = {}
        self.calculation_history: List[Dict[str, Any]] = []

        # Configuration
        self.pnl_config: Dict[str, Any] = self.config.get("pnl_loss_functions", {})
        self.calculation_interval: int = self.pnl_config.get(
            "calculation_interval",
            3600,
        )
        self.max_calculation_history: int = self.pnl_config.get(
            "max_calculation_history",
            100,
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid PnL configuration"),
            AttributeError: (False, "Missing required PnL parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="base initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the component."""
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}...")
            await self._load_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration")
                return False
            self.logger.info(f"✅ {self.__class__.__name__} initialization completed successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ {self.__class__.__name__} initialization failed: {e}")
            return False

    async def _load_configuration(self) -> None:
        """Load configuration (to be overridden by subclasses)."""

    def _validate_configuration(self) -> bool:
        """Validate configuration (to be overridden by subclasses)."""
        return True

    @handles_errors(fallback=None)
    def _update_calculation_history(self) -> None:
        """Update calculation history."""
        try:
            now = datetime.now()
            history_entry = {
                "timestamp": now.isoformat(),
                "results": self.calculation_results.copy(),
                "is_calculating": self.is_calculating,
            }
            self.calculation_history.append(history_entry)
            if len(self.calculation_history) > self.max_calculation_history:
                self.calculation_history.pop(0)
        except Exception as e:
            self.logger.exception(f"Error updating calculation history: {e}")

    @handles_errors(fallback=[])
    def get_calculation_history(self, limit: int | None = None) -> List[Dict[str, Any]]:
        """
        Get calculation history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of calculation history entries
        """
        try:
            history = self.calculation_history.copy()
            if limit:
                history = history[-limit:]
            return history
        except Exception as e:
            self.logger.exception(f"Error getting calculation history: {e}")
            return []

    @handles_errors(fallback={})
    def get_calculation_status(self) -> Dict[str, Any]:
        """
        Get current calculation status.

        Returns:
            Dictionary containing current calculation status
        """
        try:
            return {
                "is_calculating": self.is_calculating,
                "last_calculation": (
                    self.calculation_history[-1]["timestamp"]
                    if self.calculation_history
                    else None
                ),
                "results": self.calculation_results.copy(),
            }
        except Exception as e:
            self.logger.exception(f"Error getting calculation status: {e}")
            return {}