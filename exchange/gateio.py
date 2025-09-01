from typing import Any

from src.interfaces.base_interfaces import MarketData

from .base_exchange import BaseExchange


class GateioExchange(BaseExchange):
    """Minimal Gateio exchange placeholder to restore syntax integrity."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)
