import logging
import json
import threading
import time
from collections import deque
from websocket import WebSocketApp

class BinanceWebsocketClient:
    """
    This class connects to the Binance WebSocket API in a separate thread
    to receive live market data (klines, order book diffs) without blocking
    the main application logic. It provides thread-safe access to the latest data.
    """
    _ENDPOINT = "wss://fstream.binance.com/stream?streams="

    def __init__(self, symbols: list):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbols = [s.lower() for s in symbols]
        self.ws = None
        self.thread = None
        
        # Thread-safe data storage
        self.data_lock = threading.Lock()
        self.latest_klines = {s: None for s in self.symbols}
        self.order_books = {s: {'bids': {}, 'asks': {}} for s in self.symbols}
        
        self.connect()

    def _get_url(self):
        kline_streams = '/'.join([f"{s}@kline_1m" for s in self.symbols])
        depth_streams = '/'.join([f"{s}@depth" for s in self.symbols])
        return f"{self._ENDPOINT}{kline_streams}/{depth_streams}"

    def connect(self):
        self.logger.info("Connecting to Binance WebSocket...")
        self.ws = WebSocketApp(self._get_url(),
                               on_message=self._on_message,
                               on_error=self._on_error,
                               on_close=self._on_close,
                               on_open=self._on_open)
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def _on_message(self, ws, message):
        data = json.loads(message)
        stream = data.get('stream')
        payload = data.get('data')

        if not stream or not payload:
            return

        with self.data_lock:
            if '@kline' in stream:
                self._handle_kline(payload)
            elif '@depth' in stream:
                self._handle_depth(payload)

    def _handle_kline(self, payload):
        symbol = payload['s'].lower()
        kline_data = {
            "timestamp": pd.to_datetime(payload['k']['t'], unit='ms'),
            "open": float(payload['k']['o']),
            "high": float(payload['k']['h']),
            "low": float(payload['k']['l']),
            "close": float(payload['k']['c']),
            "volume": float(payload['k']['v']),
            "is_closed": payload['k']['x']
        }
        self.latest_klines[symbol] = kline_data
        # self.logger.debug(f"Received kline for {symbol}: {kline_data['close']}")

    def _handle_depth(self, payload):
        symbol = payload['s'].lower()
        book = self.order_books[symbol]
        for bid in payload['b']:
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                if price in book['bids']: del book['bids'][price]
            else:
                book['bids'][price] = qty
        for ask in payload['a']:
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                if price in book['asks']: del book['asks'][price]
            else:
                book['asks'][price] = qty

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.warning("WebSocket closed. Reconnecting...")
        time.sleep(5)
        self.connect()

    def _on_open(self, ws):
        self.logger.info("WebSocket connection opened.")

    def get_latest_kline(self, symbol: str) -> dict or None:
        with self.data_lock:
            return self.latest_klines.get(symbol.lower())

    def get_order_book(self, symbol: str) -> dict or None:
        with self.data_lock:
            # Return a copy to prevent modification outside the lock
            return self.order_books.get(symbol.lower(), {}).copy()

