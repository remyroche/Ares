from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from src.interfaces.base_interfaces import IExchangeClient, MarketData


class BaseExchange(IExchangeClient, ABC):
    """
    Base class for all exchange implementations.
    Provides standardized method signatures and common functionality.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.trade_symbol = trade_symbol.upper()
        self.password = password
        self.exchange: Any | None = None  # Will be set by subclasses

    @abstractmethod
    @abstractmethod
    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw exchange data to standardized MarketData format."""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod
    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on exchange."""

    @abstractmethod
    # Additional standardized helpers
    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Best-effort leverage setter using underlying client if supported."""
        try:
            market_id = await self._get_market_id(symbol)
        except Exception:
            market_id = symbol

        if not self.exchange:
            return False

        attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
            ("set_leverage", (leverage, market_id), {}),
            ("set_leverage", (), {"leverage": leverage, "symbol": market_id}),
            ("setLeverage", (leverage, market_id), {}),
        ]

        for method, args, kwargs in attempts:
            if hasattr(self.exchange, method):
                try:
                    await getattr(self.exchange, method)(*args, **kwargs)
                    return True
                except Exception:
                    continue
        return False

    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Best-effort margin mode setter using underlying client if supported."""
        try:
            market_id = await self._get_market_id(symbol)
        except Exception:
            market_id = symbol

        if not self.exchange:
            return False

        attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
            ("set_margin_mode", (mode, market_id), {}),
            ("set_margin_mode", (), {"marginMode": mode, "symbol": market_id}),
            ("setMarginMode", (mode, market_id), {}),
        ]

        for method, args, kwargs in attempts:
            if hasattr(self.exchange, method):
                try:
                    await getattr(self.exchange, method)(*args, **kwargs)
                    return True
                except Exception:
                    continue
        return False

    async def close(self) -> None:
        """Close the exchange connection if supported by underlying client."""
        if self.exchange and hasattr(self.exchange, "close"):
            await self.exchange.close()

    # --- Optional streaming hooks (to be implemented by subclasses as needed) ---
    # --- Convenience polling helpers ---
    # --- Default CCXT-based helpers (can be overridden by subclasses) ---