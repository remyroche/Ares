
import asyncio
import json
import logging
import time
import hmac
import hashlib
from typing import Callable

import pandas as pd
import websockets

from src.utils.logger import system_logger


class GateioWebsocketClient:
    """
    Connects to the Gate.io WebSocket API for live market data using the `websockets` library.
    Handles futures data (klines, order book) asynchronously.
    """

    _ENDPOINT = "wss://fx-ws.gateio.ws/v4/ws/usdt"

    def __init__(self, api_key: str, api_secret: str, symbols: list):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = [s.upper().replace("USDT", "_USDT") for s in symbols]
        self.ws = None

        # Asynchronous data storage
        self.latest_klines = {s: None for s in self.symbols}
        self.order_books = {s: {"bids": {}, "asks": {}} for s in self.symbols}

    async def _get_auth(self):
        """Generate authentication signature for private channels."""
        timestamp = str(int(time.time()))
        signature_string = f"channel=futures.orders&event=subscribe&time={timestamp}"
        signature = hmac.new(self.api_secret.encode('utf-8'), signature_string.encode('utf-8'), hashlib.sha512).hexdigest()
        return {
            "method": "api_key",
            "KEY": self.api_key,
            "SIGN": signature,
            "time": timestamp,
        }

    async def connect(self):
        """Establishes and maintains the WebSocket connection."""
        self.logger.info("Connecting to Gate.io WebSocket...")
        while True:
            try:
                async with websockets.connect(self._ENDPOINT) as ws:
                    self.ws = ws
                    await self._on_open()
                    async for message in ws:
                        await self._on_message(message)
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"Gate.io WebSocket closed: {e}. Reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Gate.io WebSocket error: {e}. Reconnecting...")
                await asyncio.sleep(10)

    async def _on_open(self):
        self.logger.info("Gate.io WebSocket connection opened.")
        kline_streams = [f"futures.candlesticks,1m,{s}" for s in self.symbols]
        depth_streams = [f"futures.order_book,{s},20,100ms" for s in self.symbols]

        for stream in kline_streams + depth_streams:
            await self.ws.send(json.dumps({
                "time": int(time.time()),
                "channel": stream.split(',')[0],
                "event": "subscribe",
                "payload": [stream.split(',')[1], stream.split(',')[2]] if len(stream.split(',')) > 2 else [stream.split(',')[1]]
            }))

    async def _on_message(self, message):
        data = json.loads(message)
        channel = data.get("channel")
        if not channel:
            return

        if "candlesticks" in channel:
            await self._handle_kline(data)
        elif "order_book" in channel:
            await self._handle_depth(data)

    async def _handle_kline(self, data):
        kline_data = data["result"][-1]
        symbol = kline_data['n'].lower()
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
        symbol = data["result"]['s'].lower()
        book = self.order_books[symbol]
        book['bids'] = {float(p): float(s) for p, s in data["result"]['b']}
        book['asks'] = {float(p): float(s) for p, s in data["result"]['a']}

    def get_latest_kline(self, symbol: str) -> dict or None:
        return self.latest_klines.get(symbol.lower().replace("usdt", "_usdt"))

    def get_order_book(self, symbol: str) -> dict or None:
        return self.order_books.get(symbol.lower().replace("usdt", "_usdt"), {}).copy()
