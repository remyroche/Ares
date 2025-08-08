# exchange/base_exchange.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Awaitable

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
    ):
        """
        Initialize base exchange.

        Args:
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            trade_symbol: Trading symbol (e.g., 'BTCUSDT')
            password: Optional password for exchanges that require it (e.g., OKX)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.trade_symbol = trade_symbol.upper()
        self.password = password
        self.exchange = None  # Will be set by subclasses

    @abstractmethod
    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client. Must be implemented by subclasses."""

    @abstractmethod
    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """
        Convert raw exchange data to standardized MarketData format.
        Must be implemented by subclasses.
        """

    @abstractmethod
    async def _get_market_id(self, symbol: str) -> str:
        """
        Get the market ID for a given symbol.
        Must be implemented by subclasses.
        """

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[MarketData]:
        """
        Get historical kline data in standardized format.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to retrieve

        Returns:
            List of MarketData objects
        """
        raw_data = await self._get_klines_raw(symbol, interval, limit)
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @abstractmethod
    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Get raw kline data from exchange.
        Must be implemented by subclasses.
        """

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary containing account information
        """
        return await self._get_account_info_raw()

    @abstractmethod
    async def _get_account_info_raw(self) -> dict[str, Any]:
        """
        Get raw account information from exchange.
        Must be implemented by subclasses.
        """

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """
        Create a trading order.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Order price (optional for market orders)
            order_type: Order type ('MARKET', 'LIMIT', etc.)

        Returns:
            Dictionary containing order information
        """
        return await self._create_order_raw(symbol, side, order_type, quantity, price)

    @abstractmethod
    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create raw order on exchange.
        Must be implemented by subclasses.
        """

    async def get_position_risk(self, symbol: str) -> dict[str, Any]:
        """
        Get position risk information.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary containing position risk information
        """
        return await self._get_position_risk_raw(symbol)

    @abstractmethod
    async def _get_position_risk_raw(
        self,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        Get raw position risk information from exchange.
        Must be implemented by subclasses.
        """

    # Additional standardized methods that are commonly used

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[MarketData]:
        """
        Get historical kline data for a specific time range.

        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of candles

        Returns:
            List of MarketData objects
        """
        raw_data = await self._get_historical_klines_raw(
            symbol,
            interval,
            start_time_ms,
            end_time_ms,
            limit,
        )
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @abstractmethod
    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Get raw historical kline data from exchange.
        Must be implemented by subclasses.
        """

    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get historical aggregated trades.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            limit: Maximum number of trades

        Returns:
            List of trade dictionaries
        """
        return await self._get_historical_agg_trades_raw(
            symbol,
            start_time_ms,
            end_time_ms,
            limit,
        )

    @abstractmethod
    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        Get raw historical aggregated trades from exchange.
        Must be implemented by subclasses.
        """

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Optional trading symbol filter

        Returns:
            List of open orders
        """
        return await self._get_open_orders_raw(symbol)

    @abstractmethod
    async def _get_open_orders_raw(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get raw open orders from exchange.
        Must be implemented by subclasses.
        """

    async def cancel_order(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            Dictionary containing cancellation result
        """
        return await self._cancel_order_raw(symbol, order_id)

    @abstractmethod
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """
        Cancel raw order on exchange.
        Must be implemented by subclasses.
        """

    async def get_order_status(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """
        Get order status.

        Args:
            symbol: Trading symbol
            order_id: Order ID

        Returns:
            Dictionary containing order status
        """
        return await self._get_order_status_raw(symbol, order_id)

    @abstractmethod
    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """
        Get raw order status from exchange.
        Must be implemented by subclasses.
        """

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Best-effort leverage setter using underlying client if supported."""
        try:
            market_id = await self._get_market_id(symbol)
        except Exception:
            market_id = symbol

        if not self.exchange:
            return False

        attempts = [
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
                    # Intentionally continue to try next known signature
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

        attempts = [
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
        """
        Close the exchange connection.
        """
        if self.exchange and hasattr(self.exchange, "close"):
            await self.exchange.close()

    def _convert_timestamp(self, timestamp: Any) -> datetime:
        """
        Convert exchange timestamp to datetime.

        Args:
            timestamp: Exchange timestamp (could be int, float, or string)

        Returns:
            datetime object
        """
        if isinstance(timestamp, (int, float)):
            # Assume milliseconds if timestamp is large
            if timestamp > 1e10:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        if isinstance(timestamp, str):
            # Try to parse as ISO format
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                # Try other common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse timestamp: {timestamp}")
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

    # --- Optional streaming hooks (to be implemented by subclasses as needed) ---
    async def subscribe_trades(self, symbol: str, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to live trades for symbol and invoke callback(trade_dict)."""
        raise NotImplementedError

    async def subscribe_ticker(self, symbol: str, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to live ticker/mark price updates and invoke callback(ticker_dict)."""
        raise NotImplementedError

    async def subscribe_order_book(self, symbol: str, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to live order book updates and invoke callback(book_dict)."""
        raise NotImplementedError

    # --- Convenience polling helpers ---
    async def fetch_price(self, symbol: str) -> float | None:
        """Fetch current price using ticker or order book mid as fallback."""
        try:
            # Prefer a direct ticker if subclass implements get_ticker
            if hasattr(self, "get_ticker"):
                ticker = await getattr(self, "get_ticker")(symbol)
                if ticker:
                    last = ticker.get("last") or ticker.get("mark") or ticker.get("close")
                    if last is not None:
                        return float(last)
                    # derive mid if possible
                    bid = ticker.get("bid")
                    ask = ticker.get("ask")
                    if bid is not None and ask is not None:
                        return (float(bid) + float(ask)) / 2.0
            # Fallback to order book mid
            if hasattr(self, "get_order_book"):
                book = await getattr(self, "get_order_book")(symbol, 5)
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                best_bid = float(bids[0][0]) if bids else None
                best_ask = float(asks[0][0]) if asks else None
                if best_bid is not None and best_ask is not None:
                    return (best_bid + best_ask) / 2.0
                if best_bid is not None:
                    return best_bid
                if best_ask is not None:
                    return best_ask
        except Exception:
            return None
        return None

    async def get_liquidation_price(self, symbol: str) -> float | None:
        """Best-effort liquidation price for current position on symbol."""
        try:
            risk = await self.get_position_risk(symbol)
            # Try common ccxt fields
            if isinstance(risk, list) and risk:
                # Find matching symbol
                for p in risk:
                    inst = p.get("symbol") or p.get("info", {}).get("symbol")
                    if inst and inst.replace("-", "").replace("_", "").upper().startswith(symbol.upper().replace("USDT", "")):
                        liq = p.get("liquidationPrice") or p.get("liqPrice") or p.get("liquidation_price")
                        if liq:
                            return float(liq)
                # Otherwise take first
                p = risk[0]
                liq = p.get("liquidationPrice") or p.get("liqPrice") or p.get("liquidation_price")
                if liq:
                    return float(liq)
        except Exception:
            return None
        return None

    # --- Default CCXT-based helpers (can be overridden by subclasses) ---
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Default ticker fetch using ccxt."""
        try:
            if not self.exchange:
                return {}
            market_id = await self._get_market_id(symbol) if symbol else None
            if market_id:
                return await self.exchange.fetch_ticker(market_id)
            # All tickers fallback
            tickers = await self.exchange.fetch_tickers()
            return tickers or {}
        except Exception:
            return {}

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Default order book fetch using ccxt."""
        try:
            if not self.exchange:
                return {}
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order_book(market_id, limit)
        except Exception:
            return {}
