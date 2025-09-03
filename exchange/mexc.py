from typing import Any

from src.interfaces.base_interfaces import MarketData

from .base_exchange import BaseExchange


class MexcExchange(BaseExchange):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)

    async def _initialize_exchange(self) -> None:
        self.exchange = None

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        return []

    async def _get_market_id(self, symbol: str) -> str:
        return symbol

    async def _get_klines_raw(self, symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        return {}

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        return {}

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        return []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        return {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        return {}
