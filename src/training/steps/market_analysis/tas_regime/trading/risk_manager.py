"""Risk management module for the TAS trading engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def _side_to_str(side: object) -> str:
    if hasattr(side, "value"):
        return str(getattr(side, "value"))
    return str(side)


@dataclass
class RiskConfig:
    """Configuration values for :class:`RiskManager`."""

    max_position_fraction: float = 0.25
    max_trade_fraction: float = 0.1
    min_confidence: float = 0.5
    min_trade_quantity: float = 0.001
    max_signal_value: Optional[float] = None
    default_price: float = 100.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be within (0, 1]")
        if not 0 < self.max_trade_fraction <= 1:
            raise ValueError("max_trade_fraction must be within (0, 1]")
        if self.min_confidence <= 0 or self.min_confidence > 1:
            raise ValueError("min_confidence must be within (0, 1]")
        if self.min_trade_quantity <= 0:
            raise ValueError("min_trade_quantity must be positive")


class RiskManager:
    """Basic risk checks used by :class:`TradingEngine`."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions: Dict[str, float] = {}
        self.last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API used by the engine
    # ------------------------------------------------------------------
    def check_trade_risk(
        self, symbol: str, side: object, quantity: float, current_capital: float
    ) -> bool:
        """Return ``True`` if executing the trade is within limits."""

        if quantity <= 0:
            self.logger.debug("Rejected trade for %s due to non-positive quantity", symbol)
            return False

        side_value = _side_to_str(side).lower()
        if side_value not in {"buy", "sell"}:
            self.logger.debug("Unknown order side %s", side_value)
            return False

        price = self.last_prices.get(symbol, self.config.default_price)
        trade_value = abs(quantity) * price

        if side_value == "buy" and current_capital <= 0:
            self.logger.debug("Insufficient capital for buy trade")
            return False

        max_trade_value = current_capital * self.config.max_trade_fraction
        if max_trade_value > 0 and trade_value > max_trade_value:
            self.logger.debug(
                "Trade value %.2f exceeds max allowed %.2f", trade_value, max_trade_value
            )
            return False

        current_position = self.positions.get(symbol, 0.0)
        projected_position = current_position + (quantity if side_value == "buy" else -quantity)
        if not self.config.allow_short and projected_position < 0:
            self.logger.debug("Short positions disabled – rejecting trade")
            return False

        exposure = abs(projected_position) * price
        max_exposure = current_capital * self.config.max_position_fraction
        if max_exposure > 0 and exposure > max_exposure:
            self.logger.debug(
                "Exposure %.2f exceeds max allowed %.2f", exposure, max_exposure
            )
            return False

        return True

    def check_signal_risk(self, signal: Dict[str, object], current_capital: float) -> bool:
        """Validate signal before execution."""

        symbol = str(signal.get("symbol", "")) or self.config.default_price
        price = float(signal.get("price", self.config.default_price))
        quantity = float(signal.get("quantity", 0.0))
        side = _side_to_str(signal.get("side", "buy")).lower()
        confidence = float(signal.get("confidence", 1.0))

        if confidence < self.config.min_confidence:
            self.logger.debug(
                "Signal confidence %.3f below threshold %.3f",
                confidence,
                self.config.min_confidence,
            )
            return False

        if quantity < self.config.min_trade_quantity:
            self.logger.debug(
                "Signal quantity %.6f below minimum %.6f",
                quantity,
                self.config.min_trade_quantity,
            )
            return False

        trade_value = price * abs(quantity)
        if self.config.max_signal_value and trade_value > self.config.max_signal_value:
            self.logger.debug(
                "Signal value %.2f exceeds limit %.2f",
                trade_value,
                self.config.max_signal_value,
            )
            return False

        max_trade_value = current_capital * self.config.max_trade_fraction
        if max_trade_value > 0 and trade_value > max_trade_value:
            self.logger.debug(
                "Signal value %.2f exceeds capital allocation %.2f",
                trade_value,
                max_trade_value,
            )
            return False

        current_position = float(signal.get("current_position", self.positions.get(symbol, 0.0)))
        projected_position = current_position + (quantity if side == "buy" else -quantity)
        if not self.config.allow_short and projected_position < 0:
            self.logger.debug("Short positions disabled – rejecting signal")
            return False

        exposure = abs(projected_position) * price
        max_exposure = current_capital * self.config.max_position_fraction
        if max_exposure > 0 and exposure > max_exposure:
            self.logger.debug(
                "Projected exposure %.2f exceeds limit %.2f",
                exposure,
                max_exposure,
            )
            return False

        self.last_prices[str(symbol)] = price
        return True

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def update_position(self, symbol: str, quantity: float) -> None:
        if abs(quantity) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = quantity
        self.logger.debug("Risk exposure updated %s -> %.6f", symbol, quantity)

    def reset(self) -> None:
        self.positions.clear()
        self.last_prices.clear()
        self.logger.debug("Risk manager state reset")
