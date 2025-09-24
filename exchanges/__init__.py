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
    "ResponseHandler"
]