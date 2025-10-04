"""Position management utilities for the TAS trading engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionConfig:
    """Configuration for :class:`PositionManager`."""

    max_position_fraction: float = 0.2
    min_trade_quantity: float = 0.001
    allow_short: bool = True
    default_symbol: str = "TAS"

    def __post_init__(self) -> None:
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be within (0, 1]")
        if self.min_trade_quantity <= 0:
            raise ValueError("min_trade_quantity must be positive")


class PositionManager:
    """Maintain portfolio positions and sizing logic."""

    def __init__(self, config: PositionConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Position sizing helpers
    # ------------------------------------------------------------------
    def size_from_signal(
        self,
        symbol: str,
        desired_quantity: float,
        current_capital: float,
        price: float,
    ) -> float:
        """Return the feasible quantity for a trade given constraints."""

        quantity = max(desired_quantity, self.config.min_trade_quantity)
        if price <= 0 or current_capital <= 0:
            return quantity

        max_position_value = current_capital * self.config.max_position_fraction
        current_position = abs(self.positions.get(symbol, 0.0)) * price
        available_value = max(0.0, max_position_value - current_position)
        if available_value == 0:
            return 0.0

        feasible_quantity = min(quantity, available_value / price)
        if feasible_quantity < self.config.min_trade_quantity:
            return 0.0
        return feasible_quantity

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------
    def update_position(self, symbol: str, delta: float) -> None:
        """Update stored position size for ``symbol``."""

        current = self.positions.get(symbol, 0.0)
        new_position = current + delta
        if abs(new_position) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_position
        self.logger.debug("Position updated %s: %.6f -> %.6f", symbol, current, new_position)

    def set_position(self, symbol: str, quantity: float) -> None:
        """Explicitly set position size for ``symbol``."""

        if abs(quantity) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = float(quantity)
        self.logger.debug("Position set %s: %.6f", symbol, quantity)

    def get_position(self, symbol: str) -> float:
        """Return the current position for ``symbol``."""

        return float(self.positions.get(symbol, 0.0))

    def reset(self) -> None:
        """Clear all tracked positions."""

        self.positions.clear()
        self.logger.debug("Positions reset")
