
import asyncio
import json
import logging
import zlib
from typing import Callable

import pandas as pd
import websockets

from src.utils.logger import system_logger


class MexcWebsocketClient:
    """
    Connects to the MEXC WebSocket API for live market data using the `websockets` library.
    Handles futures data (klines, order book) asynchronously.
    """

    _ENDPOINT = "wss://contract.mexc.com/ws"

    def __init__(self, symbols: list):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbols = [s.upper().replace("USDT", "_USDT") for s in symbols]
        self.ws = None

        # Asynchronous data storage
        self.latest_klines = {s: None for s in self.symbols}
        self.order_books = {s: {"bids": {}, "asks": {}} for s in self.symbols}

    async def connect(self):
        """Establishes and maintains the WebSocket connection."""
        self.logger.info("Connecting to MEXC WebSocket...")
        while True:
            try:
                async with websockets.connect(self._ENDPOINT) as ws:
                    self.ws = ws
                    await self._on_open()
                    async for message in ws:
                        await self._on_message(message)
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"MEXC WebSocket closed: {e}. Reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"MEXC WebSocket error: {e}. Reconnecting...")
                await asyncio.sleep(10)

    async def _on_open(self):
        self.logger.info("MEXC WebSocket connection opened.")
        for symbol in self.symbols:
            await self.ws.send(json.dumps({"method": "sub.kline", "param": {"symbol": symbol, "interval": "Min1"}}))
            await self.ws.send(json.dumps({"method": "sub.depth", "param": {"symbol": symbol, "depth": 20}}))

    async def _on_message(self, message):
        data = json.loads(zlib.decompress(message, 16 + zlib.MAX_WBITS).decode('utf-8'))
        channel = data.get("channel")
        if not channel:
            return

        if channel == "push.kline":
            await self._handle_kline(data)
        elif channel == "push.depth":
            await self._handle_depth(data)

    async def _handle_kline(self, data):
        symbol = data["symbol"].lower()
        kline_data = data["data"]
        processed_kline = {
            "timestamp": pd.to_datetime(kline_data['t'], unit="s"),
            "open": float(kline_data['o']),
            "high": float(kline_data['h']),
            "low": float(kline_data['l']),
            "close": float(kline_data['c']),
            "volume": float(kline_data['v']),
            "is_closed": True,
        }
        self.latest_klines[symbol] = processed_kline

    async def _handle_depth(self, data):
        symbol = data["symbol"].lower()
        book = self.order_books[symbol]
        if "bids" in data["data"]:
            book['bids'] = {float(p): float(v) for p, v in data["data"]["bids"]}
        if "asks" in data["data"]:
            book['asks'] = {float(p): float(v) for p, v in data["data"]["asks"]}

    def get_latest_kline(self, symbol: str) -> dict or None:
        return self.latest_klines.get(symbol.lower().replace("usdt", "_usdt"))

    def get_order_book(self, symbol: str) -> dict or None:
        return self.order_books.get(symbol.lower().replace("usdt", "_usdt"), {}).copy()
