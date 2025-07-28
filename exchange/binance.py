import time
import requests
import hmac
import hashlib
import logging
import threading
import json
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from websocket import WebSocketApp, enableTrace
import collections # For deque

# Import the main CONFIG dictionary
from config import CONFIG

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable WebSocket tracing for debugging (optional)
# enableTrace(True)

class BinanceFuturesAPI:
    # Base URLs are now accessed from CONFIG
    # BASE_URL_MAINNET = "https://fapi.binance.com"
    # WS_BASE_URL_MAINNET = "wss://fstream.binance.com"
    # BASE_URL_TESTNET = "https://testnet.binancefuture.com"
    # WS_BASE_URL_TESTNET = "wss://stream.binancefuture.com"

    RATE_LIMIT_SLEEP = 1.1  # seconds

    def __init__(self, api_key, api_secret, testnet=True, symbol=None, interval=None, config=CONFIG): # Default config to global CONFIG
        self.API_KEY = api_key
        self.API_SECRET = api_secret
        self.testnet = testnet
        self.symbol = symbol if symbol else config['SYMBOL'] # Use provided symbol or from CONFIG
        self.interval = interval if interval else config['INTERVAL'] # Use provided interval or from CONFIG
        self.config = config # Use the global CONFIG dictionary

        # Access base URLs from config
        if self.testnet:
            self.BASE_URL = self.config['live_trading'].get('testnet_rest_url', "https://testnet.binancefuture.com")
            self.WS_BASE_URL = self.config['live_trading'].get('testnet_ws_url', "wss://stream.binancefuture.com")
            logger.info("Using Binance Futures TESTNET environment.")
        else:
            self.BASE_URL = self.config['live_trading'].get('mainnet_rest_url', "https://fapi.binance.com")
            self.WS_BASE_URL = self.config['live_trading'].get('mainnet_ws_url', "wss://fstream.binance.com")
            logger.info("Using Binance Futures MAINNET environment.")

        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.API_KEY})

        retries = Retry(total=5, backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # WebSocket related attributes
        self.ws_clients = {} # Stores WebSocketApp instances
        self.ws_threads = {} # Stores WebSocket threads
        self.listen_key = None # For user data stream
        self.listen_key_keep_alive_thread = None # To manage the keep-alive thread

        # Data buffers for WebSocket streams (using deque for efficient appends/pops)
        self.kline_buffer = collections.deque(maxlen=1000) # Store last 1000 1m klines
        self.agg_trade_buffer = collections.deque(maxlen=5000) # Store last 5000 agg trades
        self.order_book_buffer = {'bids': [], 'asks': []} # Store latest order book snapshot
        self.account_balance = {} # Store latest account balance
        self.current_position = {} # Store latest position details

        # Lock for thread-safe access to buffers
        self.data_lock = threading.Lock()

    def _sign_payload(self, payload):
        query_string = urlencode(payload)
        signature = hmac.new(self.API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    def _send_signed_request(self, method, endpoint, payload={}):
        """
        Sends a signed request to the Binance API.
        Includes enhanced error handling and retries.
        """
        for attempt in range(CONFIG['live_trading'].get('max_retries', 3)): # Use max_retries from config
            try:
                payload["timestamp"] = int(time.time() * 1000)
                signed_payload = self._sign_payload(payload)
                url = self.BASE_URL + endpoint
                
                if method == "GET":
                    response = self.session.request(method, url, params=signed_payload)
                else: # POST, PUT, DELETE
                    response = self.session.request(method, url, data=signed_payload) # Use data for POST/PUT/DELETE
                
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error for {method} {endpoint}: {http_err} - Response: {http_err.response.text}")
                if attempt < CONFIG['live_trading'].get('max_retries', 3) - 1 and http_err.response.status_code in [429, 500, 502, 503, 504]:
                    retry_delay = CONFIG['live_trading'].get('retry_delay_seconds', 5) # Use retry_delay from config
                    logger.warning(f"Retrying {method} {endpoint} in {retry_delay} seconds (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                else:
                    return {"error": str(http_err), "code": http_err.response.status_code, "msg": http_err.response.text}
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Request error for {method} {endpoint}: {req_err}")
                if attempt < CONFIG['live_trading'].get('max_retries', 3) - 1:
                    retry_delay = CONFIG['live_trading'].get('retry_delay_seconds', 5)
                    logger.warning(f"Retrying {method} {endpoint} in {retry_delay} seconds (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                else:
                    return {"error": str(req_err)}
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error for {method} {endpoint}: {json_err} - Response text: {response.text}")
                return {"error": f"JSON decode error: {json_err}", "response_text": response.text}
        return {"error": f"Failed to complete {method} {endpoint} after multiple retries."}

    def _send_public_request(self, endpoint, payload={}):
        """
        Sends a public request to the Binance API.
        Includes enhanced error handling and retries.
        """
        for attempt in range(CONFIG['live_trading'].get('max_retries', 3)): # Use max_retries from config
            try:
                url = self.BASE_URL + endpoint
                response = self.session.get(url, params=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error for public {endpoint}: {http_err} - Response: {http_err.response.text}")
                if attempt < CONFIG['live_trading'].get('max_retries', 3) - 1 and http_err.response.status_code in [429, 500, 502, 503, 504]:
                    retry_delay = CONFIG['live_trading'].get('retry_delay_seconds', 5)
                    logger.warning(f"Retrying public {endpoint} in {retry_delay} seconds (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                else:
                    return {"error": str(http_err), "code": http_err.response.status_code, "msg": http_err.response.text}
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Request error for public {endpoint}: {req_err}")
                if attempt < CONFIG['live_trading'].get('max_retries', 3) - 1:
                    retry_delay = CONFIG['live_trading'].get('retry_delay_seconds', 5)
                    logger.warning(f"Retrying public {endpoint} in {retry_delay} seconds (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                else:
                    return {"error": str(req_err)}
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error for public {endpoint}: {json_err} - Response text: {response.text}")
                return {"error": f"JSON decode error: {json_err}", "response_text": response.text}
        return {"error": f"Failed to complete public {endpoint} after multiple retries."}

    # --- REST API Functions ---
    def get_server_time(self):
        """Get server time to check API connectivity."""
        return self._send_public_request("/fapi/v1/time")

    def get_exchange_info(self):
        """Get exchange information (symbols, filters, etc.)."""
        return self._send_public_request("/fapi/v1/exchangeInfo")

    def get_account_balance(self):
        """Get account balance and asset information."""
        return self._send_signed_request("GET", "/fapi/v2/balance")

    def get_position_risk(self, symbol=None):
        """Get current position risk for all symbols or a specific symbol."""
        payload = {"symbol": symbol.upper()} if symbol else {}
        return self._send_signed_request("GET", "/fapi/v2/positionRisk", payload)

    def change_leverage(self, symbol, leverage):
        """Change leverage for a specific symbol."""
        return self._send_signed_request("POST", "/fapi/v1/leverage", {
            "symbol": symbol.upper(),
            "leverage": leverage
        })

    def place_order(self, symbol, side, type, quantity, price=None, stop_price=None, time_in_force="GTC", reduce_only=False, new_client_order_id=None):
        """Place a new order."""
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": type.upper(),
            "quantity": quantity,
            "newOrderRespType": "RESULT" # Ensure full response
        }
        if price: payload["price"] = price
        if stop_price: payload["stopPrice"] = stop_price
        if type.upper() in ["LIMIT", "STOP", "TAKE_PROFIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"]:
            payload["timeInForce"] = time_in_force
        if reduce_only: payload["reduceOnly"] = "true" # Binance API expects string "true"
        if new_client_order_id: payload["newClientOrderId"] = new_client_order_id

        return self._send_signed_request("POST", "/fapi/v1/order", payload)

    def cancel_order(self, symbol, order_id=None, orig_client_order_id=None):
        """Cancel an open order."""
        payload = {"symbol": symbol.upper()}
        if order_id: payload["orderId"] = order_id
        if orig_client_order_id: payload["origClientOrderId"] = orig_client_order_id
        return self._send_signed_request("DELETE", "/fapi/v1/order", payload)

    def cancel_all_open_orders(self, symbol):
        """Cancel all open orders for a specific symbol."""
        return self._send_signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()})

    # --- WebSocket Stream Handlers ---
    def _on_open(self, ws):
        logger.info(f"WebSocket opened: {ws.url}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed: {ws.url} - Status: {close_status_code}, Message: {close_msg}")
        # Attempt to reconnect if closed unexpectedly
        stream_name = next((name for name, client in self.ws_clients.items() if client == ws), None)
        if stream_name:
            logger.info(f"Attempting to restart WebSocket stream: {stream_name}")
            # Remove old client and thread references
            if stream_name in self.ws_clients:
                del self.ws_clients[stream_name]
            if stream_name in self.ws_threads:
                del self.ws_threads[stream_name]
            # Restart the stream
            if stream_name == self.listen_key: # User data stream
                self.start_user_data_stream()
            else: # Public data stream
                # Need to map stream_name back to its type (kline, aggTrade, depth)
                # This is a bit complex, might need to store stream type in ws_clients dict
                # For simplicity, if it's a public stream, try to restart all public streams
                self.start_data_streams() # This will attempt to restart all public streams

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {ws.url} - {error}")
        # The _on_close handler will be called after an error, which can handle reconnection.

    def _on_message_kline(self, ws, message):
        data = json.loads(message)['k']
        kline = {
            'open_time': data['t'],
            'open': float(data['o']),
            'high': float(data['h']),
            'low': float(data['l']),
            'close': float(data['c']),
            'volume': float(data['v']),
            'close_time': data['T'],
            'quote_asset_volume': float(data['q']),
            'number_of_trades': int(data['n']),
            'taker_buy_base_asset_volume': float(data['V']),
            'taker_buy_quote_asset_volume': float(data['Q']),
            'ignore': float(data['B'])
        }
        with self.data_lock:
            self.kline_buffer.append(kline)

    def _on_message_agg_trade(self, ws, message):
        data = json.loads(message)['data']
        agg_trade = {
            'a': data['a'], # Aggregate tradeId
            'p': float(data['p']), # Price
            'q': float(data['q']), # Quantity
            'f': data['f'], # First tradeId
            'l': data['l'], # Last tradeId
            'T': data['T'], # Timestamp
            'm': data['m'], # Was the buyer the maker?
            'M': data['M']  # Was the trade the best price match?
        }
        with self.data_lock:
            self.agg_trade_buffer.append(agg_trade)

    def _on_message_depth(self, ws, message):
        data = json.loads(message)
        with self.data_lock:
            self.order_book_buffer['bids'] = [[float(p), float(q)] for p, q in data['b']]
            self.order_book_buffer['asks'] = [[float(p), float(q)] for p, q in data['a']]

    def _on_message_user_data(self, ws, message):
        data = json.loads(message)
        event_type = data['e']
        
        with self.data_lock:
            if event_type == 'ACCOUNT_UPDATE':
                # Update balances
                for asset_info in data['a']['B']:
                    asset_name = asset_info['a']
                    self.account_balance[asset_name] = {
                        'free': float(asset_info['f']),
                        'locked': float(asset_info['l'])
                    }
                # Update positions
                for position_info in data['a']['P']:
                    symbol = position_info['s']
                    self.current_position[symbol] = {
                        'positionAmt': float(position_info['pa']),
                        'entryPrice': float(position_info['ep']),
                        'unrealizedPnl': float(position_info['up']),
                        'liquidationPrice': float(position_info['liqP']),
                        'leverage': int(position_info['leverage'])
                    }
                logger.info(f"User Data: Account/Position Update. Balance: {self.account_balance.get('USDT',{})}, Position: {self.current_position.get(self.symbol, {})}")

            elif event_type == 'ORDER_TRADE_UPDATE':
                order_data = data['o']
                logger.info(f"User Data: Order Update - Symbol: {order_data['s']}, Status: {order_data['X']}, Side: {order_data['S']}, Price: {order_data['p']}, Quantity: {order_data['q']}")
                # You might want to store order history or update local order status here
            else:
                logger.debug(f"User Data: Unhandled event type: {event_type}")

    def _start_websocket(self, stream_name, on_message_handler):
        """Starts a single WebSocket stream in a new thread."""
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        ws = WebSocketApp(url,
                          on_open=self._on_open,
                          on_message=on_message_handler,
                          on_error=self._on_error,
                          on_close=self._on_close)
        
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        self.ws_clients[stream_name] = ws
        self.ws_threads[stream_name] = thread
        logger.info(f"Started WebSocket stream: {stream_name}")
        return ws

    def start_data_streams(self):
        """Starts all configured public data WebSocket streams."""
        ws_config = self.config.get("live_trading", {}).get("websocket_streams", {})
        
        # Kline Stream
        kline_stream_name = ws_config.get("kline", f"{self.symbol.lower()}@kline_{self.interval}")
        if kline_stream_name not in self.ws_clients: # Prevent re-starting if already running
            self._start_websocket(kline_stream_name, self._on_message_kline)

        # Aggregated Trade Stream
        agg_trade_stream_name = ws_config.get("aggTrade", f"{self.symbol.lower()}@aggTrade")
        if agg_trade_stream_name not in self.ws_clients: # Prevent re-starting if already running
            self._start_websocket(agg_trade_stream_name, self._on_message_agg_trade)

        # Order Book Depth Stream
        depth_stream_name = ws_config.get("depth", f"{self.symbol.lower()}@depth5@100ms")
        if depth_stream_name not in self.ws_clients: # Prevent re-starting if already running
            self._start_websocket(depth_stream_name, self._on_message_depth)

    def start_user_data_stream(self):
        """Starts the user data stream (requires listen key)."""
        if not self.API_KEY or not self.API_SECRET:
            logger.warning("API Key/Secret not provided. Cannot start user data stream.")
            return

        # Stop existing keep-alive thread if it's running
        if self.listen_key_keep_alive_thread and self.listen_key_keep_alive_thread.is_alive():
            # No direct way to stop a thread, but we can signal it to stop
            # For simplicity, we'll just let the old thread die and start a new one.
            # A more robust solution would involve an Event object for graceful shutdown.
            logger.info("Stopping existing listenKey keep-alive thread.")
            # self.listen_key_keep_alive_thread.join(timeout=1) # Try to join briefly
            self.listen_key_keep_alive_thread = None # Clear reference

        # Get listen key
        response = self._send_signed_request("POST", "/fapi/v1/listenKey")
        if "listenKey" in response:
            self.listen_key = response["listenKey"]
            logger.info(f"Obtained listenKey: {self.listen_key}")
            
            # Start WebSocket if not already running for this listen key
            if self.listen_key not in self.ws_clients:
                self._start_websocket(self.listen_key, self._on_message_user_data)
            else:
                logger.info(f"User data stream for listenKey {self.listen_key} already active.")
            
            # Keep listen key alive (ping every 30 minutes)
            def keep_alive():
                while True:
                    time.sleep(1800) # 30 minutes
                    response = self._send_signed_request("PUT", "/fapi/v1/listenKey")
                    if "listenKey" in response or response.get('msg') == 'The listen key will be extended to 60 minutes.':
                        logger.info("listenKey kept alive.")
                    else:
                        logger.error(f"Failed to keep listenKey alive: {response.get('msg', 'Unknown error')}. Attempting to re-obtain listenKey.")
                        # If keep-alive fails, try to restart the user data stream entirely
                        self.stop_all_streams() # Stop all streams to ensure clean state
                        self.start_data_streams() # Restart public streams
                        self.start_user_data_stream() # Re-obtain and restart user data stream
                        break # Exit this thread, a new one will be created
            
            self.listen_key_keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
            self.listen_key_keep_alive_thread.start()
        else:
            logger.error(f"Failed to obtain listenKey: {response.get('msg', 'Unknown error')}")

    def stop_all_streams(self):
        """Stops all active WebSocket connections."""
        for stream_name, ws in list(self.ws_clients.items()): # Iterate over a copy
            logger.info(f"Stopping WebSocket stream: {stream_name}")
            try:
                ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {stream_name}: {e}")
            finally:
                if stream_name in self.ws_clients:
                    del self.ws_clients[stream_name]
                if stream_name in self.ws_threads:
                    # Attempt to join the thread, though direct stopping is hard
                    thread = self.ws_threads[stream_name]
                    if thread.is_alive():
                        thread.join(timeout=1) # Give it a moment
                    del self.ws_threads[stream_name]
        self.ws_clients = {}
        self.ws_threads = {}
        self.listen_key = None # Clear listen key on full stop
        if self.listen_key_keep_alive_thread and self.listen_key_keep_alive_thread.is_alive():
            # Signal the thread to stop if possible (e.g., via an Event object)
            # For now, just clear the reference and let it eventually terminate
            self.listen_key_keep_alive_thread = None 
        logger.info("All WebSocket streams stopped.")

    # --- Data Access Methods for Pipeline ---
    def get_latest_klines(self, num_klines=100):
        """Returns the latest klines as a pandas DataFrame."""
        with self.data_lock:
            # Convert deque of dicts to list, then to DataFrame
            klines_list = list(self.kline_buffer)[-num_klines:]
            if not klines_list:
                return pd.DataFrame()
            df = pd.DataFrame(klines_list)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return df

    def get_latest_agg_trades(self, num_trades=500):
        """Returns the latest aggregated trades as a pandas DataFrame."""
        with self.data_lock:
            trades_list = list(self.agg_trade_buffer)[-num_trades:]
            if not trades_list:
                return pd.DataFrame()
            df = pd.DataFrame(trades_list)
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
            numeric_cols = ['price', 'quantity']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            return df[['price', 'quantity', 'is_buyer_maker']]

    def get_latest_order_book(self):
        """Returns the latest order book snapshot."""
        with self.data_lock:
            return self.order_book_buffer.copy()

    def get_latest_account_balance(self):
        """Returns the latest account balance snapshot."""
        with self.data_lock:
            return self.account_balance.copy()

    def get_latest_position(self, symbol=None):
        """Returns the latest position for a given symbol or all positions."""
        with self.data_lock:
            if symbol:
                return self.current_position.get(symbol.upper(), {})
            return self.current_position.copy()
