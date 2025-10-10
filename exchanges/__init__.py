"""
Exchange-Agnostic Trading Interface

This module provides exchange-agnostic interfaces for trading operations.
Enhanced with multi-exchange support and base exchange components.
"""

from .trading_receiver import TradingReceiver
from .order_router import OrderRouter
from .data_aggregator import DataAggregator
from .exchange_registry import ExchangeRegistry

# Import base exchange components
from .base_exchange import (
    BaseExchange,
    MultiExchangeBase,
    ExchangeMessageHandler,
    ExchangeResponseHandler,
    ExchangeMessageHandler as MessageHandler,
    ExchangeResponseHandler as ResponseHandler
)

# Import exchange factory
from .factory import ExchangeFactory

# Import exchange implementations
from .binance import BinanceExchange
from .bingx import BingXExchange, create_bingx_exchange
from .gateio import GateioExchange
from .mexc import MexcExchange
from .okx import OkxExchange, create_okx_exchange
from .phemex import PhemexExchange

# Import exchange dispatcher
from .exchange_dispatcher import (
    ExchangeDispatcher, 
    ExchangeConfig, 
    ExchangeType,
    create_exchange_dispatcher,
    create_okx_dispatcher,
    create_binance_dispatcher
)

__all__ = [
    "TradingReceiver",
    "OrderRouter",
    "DataAggregator",
    "ExchangeRegistry",
    "BaseExchange",
    "MultiExchangeBase",
    "ExchangeMessageHandler",
    "ExchangeResponseHandler",
    "MessageHandler",
    "ResponseHandler",
    "ExchangeFactory",
    "BinanceExchange",
    "BingXExchange",
    "create_bingx_exchange",
    "GateioExchange",
    "MexcExchange",
    "OkxExchange",
    "create_okx_exchange",
    "PhemexExchange",
    "ExchangeDispatcher",
    "ExchangeConfig",
    "ExchangeType",
    "create_exchange_dispatcher",
    "create_okx_dispatcher",
    "create_binance_dispatcher"
]