
import asyncio
import json
import logging
import time
import hmac
import base64
from typing import Callable

import pandas as pd
import websockets

from src.utils.logger import system_logger


class OkxWebsocketClient:
    """
    Connects to the OKX WebSocket API for live market data using the `websockets` library.
    Handles futures data (klines, order book) asynchronously.
    """

    _ENDPOINT = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self, api_key: str, api_secret: str, passphrase: str, symbols: list):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.symbols = [s.upper().replace("USDT", "-USDT-SWAP") for s in symbols]
        self.ws = None

        # Asynchronous data storage
        self.latest_klines = {s: None for s in self.symbols}
        self.order_books = {s: {"bids": {}, "asks": {}} for s in self.symbols}

    async def _get_auth_args(self):
        """Generate authentication arguments for private channels."""
        timestamp = str(time.time())
        message = timestamp + "GET" + "/users/self/verify"
        mac = hmac.new(bytes(self.api_secret, 'utf-8'), bytes(message, 'utf-8'), digestmod='sha256')
        sign = base64.b64encode(mac.digest()).decode()
        return [self.api_key, self.passphrase, timestamp, sign]

    async def connect(self):
        """Establishes and maintains the WebSocket connection."""
        self.logger.info("Connecting to OKX WebSocket...")
        while True:
            try:
                async with websockets.connect(self._ENDPOINT) as ws:
                    self.ws = ws
                    await self._on_open()
                    async for message in ws:
                        await self._on_message(message)
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"OKX WebSocket closed: {e}. Reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"OKX WebSocket error: {e}. Reconnecting...")
                await asyncio.sleep(10)

    async def _on_open(self):
        self.logger.info("OKX WebSocket connection opened.")
        kline_args = [{"channel": "candle1m", "instId": s} for s in self.symbols]
        depth_args = [{"channel": "books", "instId": s} for s in self.symbols]
        await self.ws.send(json.dumps({"op": "subscribe", "args": kline_args + depth_args}))

    async def _on_message(self, message):
        if message == 'pong':
            return
        data = json.loads(message)
        arg = data.get("arg")
        if not arg:
            return

        if arg["channel"] == "candle1m":
            await self._handle_kline(data)
        elif arg["channel"] == "books":
            await self._handle_depth(data)

    async def _handle_kline(self, data):
        instId = data["arg"]["instId"].lower()
        kline_data = data["data"][0]
        processed_kline = {
            "timestamp": pd.to_datetime(kline_data[0], unit="ms"),
            "open": float(kline_data[1]),
            "high": float(kline_data[2]),
            "low": float(kline_data[3]),
            "close": float(kline_data[4]),
            "volume": float(kline_data[5]),
            "is_closed": True,
        }
        self.latest_klines[instId] = processed_kline

    async def _handle_depth(self, data):
        instId = data["arg"]["instId"].lower()
        book = self.order_books[instId]
        if data["action"] == "snapshot":
            book['bids'] = {float(p): float(s) for p, s, _, _ in data["data"][0]["bids"]}
            book['asks'] = {float(p): float(s) for p, s, _, _ in data["data"][0]["asks"]}
        elif data["action"] == "update":
            for p, s, _, _ in data["data"][0]["bids"]:
                if float(s) == 0:
                    if float(p) in book['bids']:
                        del book['bids'][float(p)]
                else:
                    book['bids'][float(p)] = float(s)
            for p, s, _, _ in data["data"][0]["asks"]:
                if float(s) == 0:
                    if float(p) in book['asks']:
                        del book['asks'][float(p)]
                else:
                    book['asks'][float(p)] = float(s)

    def get_latest_kline(self, symbol: str) -> dict or None:
        return self.latest_klines.get(symbol.lower().replace("usdt", "-usdt-swap"))

    def get_order_book(self, symbol: str) -> dict or None:
        return self.order_books.get(symbol.lower().replace("usdt", "-usdt-swap"), {}).copy()
