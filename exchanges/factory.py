from typing import Any

from src.config import get_complete_config

# from .gateio import GateioExchange  # Commented out to avoid circular dependency
# from .mexc import MexcExchange  # Commented out to avoid circular dependency
# from .okx import OkxExchange  # Commented out to avoid circular dependency
# from .binance import BinanceExchange  # Commented out to avoid circular dependency
# from .phemex import PhemexExchange  # Commented out to avoid circular dependency
import logging


class ExchangeFactory:
    @staticmethod
    def get_exchange(exchange_name: str):
        name = (exchange_name or "").lower()
        cfg = get_complete_config()
        env = cfg.get("environment", {})
        exchanges_cfg = cfg.get("exchanges", {})
        ex_cfg = exchanges_cfg.get(name, {})
        symbol = env.get("trade_symbol", "BTCUSDT")

        if name == "binance":
            # Import locally to avoid circular dependency
            from .binance import BinanceExchange
            return BinanceExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
                password=str(ex_cfg.get("password", "")) or None,
            )

        if name == "okx":
            # Import locally to avoid circular dependency
            from .okx import OkxExchange
            return OkxExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
                password=str(ex_cfg.get("password", "")) or None,
            )

        if name == "gateio":
            # Import locally to avoid circular dependency
            from .gateio import GateioExchange
            return GateioExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
            )

        if name == "mexc":
            # Import locally to avoid circular dependency
            from .mexc import MexcExchange
            return MexcExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
            )

        if name == "phemex":
            return PhemexExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
                password=str(ex_cfg.get("password", "")) or None,
            )

        msg = f"Unsupported exchange: {exchange_name}"
        raise ValueError(msg)
