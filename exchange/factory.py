
from src.config import ares_config
from .base_exchange import BaseExchange
from .binance import BinanceExchange
from .gateio import GateioExchange
from .mexc import MexcExchange
from .okx import OkxExchange


class ExchangeFactory:
    @staticmethod
    def get_exchange(exchange_name: str):
        exchange_name = exchange_name.lower()
        config = ares_config.get('exchanges', {}).get(exchange_name, {})

        if exchange_name == "binance":
            return BinanceExchange(
                api_key=config.get("api_key"),
                api_secret=config.get("api_secret"),
                trade_symbol=ares_config.trade_symbol,
            )
        elif exchange_name == "okx":
            return OkxExchange(
                api_key=config.get("api_key"),
                api_secret=config.get("api_secret"),
                password=config.get("password"),
                trade_symbol=ares_config.trade_symbol,
            )
        elif exchange_name == "gateio":
            return GateioExchange(
                api_key=config.get("api_key"),
                api_secret=config.get("api_secret"),
                trade_symbol=ares_config.trade_symbol,
            )
        elif exchange_name == "mexc":
            return MexcExchange(
                api_key=config.get("api_key"),
                api_secret=config.get("api_secret"),
                trade_symbol=ares_config.trade_symbol,
            )
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
