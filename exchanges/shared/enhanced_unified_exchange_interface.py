"""
Lightweight fallback for enhanced_unified_exchange_interface to satisfy imports in
klines adapters when full trading stack is not available.
"""

from typing import Any, Optional, List
from enum import Enum


class ExchangeType(Enum):
    BINANCE = "binance"
    BINGX = "bingx"
    OKX = "okx"
    MEXC = "mexc"
    GATEIO = "gateio"
    PHEMEX = "phemex"


class EnhancedUnifiedExchangeAdapter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Any:
        return []


async def get_enhanced_standardized_klines(
    exchange_adapter: EnhancedUnifiedExchangeAdapter,
    symbol: str,
    interval: str,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    limit: Optional[int] = None,
) -> Any:
    return await exchange_adapter.get_klines(symbol, interval, start_time, end_time, limit)
