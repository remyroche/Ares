import os
from typing import Any

from src.config import get_complete_config
from src.utils.api_key_loader import get_api_keys

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

        # Check TRADE/LIVE flags to determine which keys to use
        use_live = os.getenv("TRADE") == "1" or os.getenv("LIVE") == "1"
        
        # Load API keys from secret/api_keys.json
        api_keys = get_api_keys(name, use_live=use_live)
        api_key = api_keys.get("api_key") or ex_cfg.get("api_key", "")  # Fallback to config if not in api_keys
        api_secret = api_keys.get("api_secret") or ex_cfg.get("api_secret", "")  # Fallback to config if not in api_keys
        password = api_keys.get("password") or (ex_cfg.get("password") or None)

        if name == "binance":
            # Import locally to avoid circular dependency
            from .binance import BinanceExchange
            return BinanceExchange(
                api_key=str(api_key),
                api_secret=str(api_secret),
                trade_symbol=str(symbol),
                password=password,
            )

        if name == "okx":
            # Import locally to avoid circular dependency
            from .okx import OkxExchange
            return OkxExchange(
                api_key=str(api_key),
                api_secret=str(api_secret),
                trade_symbol=str(symbol),
                password=password,
            )

        if name == "gateio":
            # Import locally to avoid circular dependency
            from .gateio import GateioExchange
            return GateioExchange(
                api_key=str(api_key),
                api_secret=str(api_secret),
                trade_symbol=str(symbol),
            )

        if name == "mexc":
            # Import locally to avoid circular dependency
            from .mexc import MexcExchange
            return MexcExchange(
                api_key=str(api_key),
                api_secret=str(api_secret),
                trade_symbol=str(symbol),
            )

        if name == "phemex":
            # Import locally to avoid circular dependency
            from .phemex import PhemexExchange
            return PhemexExchange(
                api_key=str(api_key),
                api_secret=str(api_secret),
                trade_symbol=str(symbol),
                password=password,
            )

        msg = f"Unsupported exchange: {exchange_name}"
        raise ValueError(msg)
