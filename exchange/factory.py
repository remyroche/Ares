from typing import Any

from src.config import get_complete_config

from .gateio import GateioExchange
from .mexc import MexcExchange
from .okx import OkxExchange


class ExchangeFactory:
    @staticmethod
    def get_exchange(exchange_name: str):
        name = (exchange_name or "").lower()
        cfg: dict[str, Any] = get_complete_config()
        env = cfg.get("environment", {})
        exchanges_cfg = cfg.get("exchanges", {})
        ex_cfg = exchanges_cfg.get(name, {})
        symbol = env.get("trade_symbol", "BTCUSDT")

        if name == "binance":
            # Prefer the refactored, canonical implementation
            from src.exchange.binance import BinanceExchange as CleanBinance

            return CleanBinance(cfg)

        if name == "okx":
            return OkxExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
                password=str(ex_cfg.get("password", "")) or None,
            )

        if name == "gateio":
            return GateioExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
            )

        if name == "mexc":
            return MexcExchange(
                api_key=str(ex_cfg.get("api_key", "")),
                api_secret=str(ex_cfg.get("api_secret", "")),
                trade_symbol=str(symbol),
            )

        msg = f"Unsupported exchange: {exchange_name}"
        raise ValueError(msg)
