"""
Enhanced Binance Exchange Client with CCXT Fallback

This module provides a robust Binance exchange client with comprehensive error handling,
graceful dependency management, CCXT fallback support, and full functionality for data collection and trading.

Features:
- Graceful handling of missing dependencies
- CCXT integration as fallback mechanism
- Comprehensive error recovery
- Rate limiting and retry logic
- Support for all Binance API endpoints
- Memory-efficient data processing
- Full integration with the data collection pipeline
"""

import hashlib
import hmac
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from urllib.parse import urlencode
from datetime import datetime, timedelta

# Graceful dependency handling
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    import ccxt  # noqa: F401
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None
    ccxt_async = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    # Create mock decorators
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def stop_after_attempt(*args, **kwargs):
        return None
    
    def wait_exponential(*args, **kwargs):
        return None

# Import project utilities
try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger('BinanceExchange')

try:
    from src.core.decorators import handles_errors
except ImportError:
    def handles_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from src.utils.error_recovery.advanced_error_recovery import get_error_recovery, with_error_recovery
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False
    def with_error_recovery(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(msg):
        print(msg)

logger = system_logger.getChild('BinanceExchange')

class BinanceAPIError(Exception):
    """Custom exception for Binance API errors."""
    pass

class BinanceDependencyError(Exception):
    """Exception raised when required dependencies are missing."""
    pass

class BinanceExchange:
    """
    Enhanced Binance exchange client with CCXT fallback support.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize enhanced Binance exchange.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('BinanceExchange')
        self.is_connected = False
        self.session = None
        self.ccxt_exchange = None
        
        # API endpoints
        self.base_url = 'https://api.binance.com'
        self.testnet_url = 'https://testnet.binance.vision'
        self.futures_base_url = 'https://fapi.binance.com'
        self.testnet_futures_url = 'https://testnet.binancefuture.com'
        
        # Configuration
        self.exchange_config = self.config.get('binance_exchange', {})
        self.api_key = self.exchange_config.get('api_key')
        self.api_secret = self.exchange_config.get('api_secret')
        self.use_testnet = self.exchange_config.get('use_testnet', True)
        self.timeout = self.exchange_config.get('timeout', 30)
        self.max_retries = self.exchange_config.get('max_retries', 3)
        
        # Rate limiting
        self.rate_limit_enabled = self.exchange_config.get('rate_limit_enabled', True)
        self.rate_limit_requests = self.exchange_config.get('rate_limit_requests', 1200)
        self.rate_limit_window = self.exchange_config.get('rate_limit_window', 60)
        self.request_times = []
        
        # API fallback configuration
        self.use_ccxt_fallback = self.exchange_config.get('use_ccxt_fallback', True)
        self.primary_api_failed = False
        
        # Error recovery
        if ERROR_RECOVERY_AVAILABLE:
            self.error_recovery = get_error_recovery()
        else:
            self.error_recovery = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'ccxt_fallback_used': 0,
            'last_request_time': None
        }
        
        self.logger.info("🚀 Enhanced Binance Exchange initialized")
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for required dependencies and provide helpful error messages."""
        missing_deps = []
        
        if not AIOHTTP_AVAILABLE:
            missing_deps.append("aiohttp (for HTTP requests)")
        
        if not CCXT_AVAILABLE:
            missing_deps.append("ccxt (for fallback API)")
        
        if missing_deps:
            self.logger.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            self.logger.warning("   Some functionality may be limited")
            self.logger.warning("   Install with: pip install aiohttp ccxt pandas numpy tenacity")
        else:
            self.logger.info("✅ All dependencies available")

    def _get_base_url(self) -> str:
        """Get base URL based on testnet setting."""
        return self.testnet_url if self.use_testnet else self.base_url

    def _get_futures_base_url(self) -> str:
        """Get futures base URL based on testnet setting."""
        return self.testnet_futures_url if self.use_testnet else self.futures_base_url

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC signature for authenticated requests.

        Args:
            params: Request parameters

        Returns:
            str: HMAC signature
        """
        try:
            if not self.api_secret:
                raise ValueError("API secret not configured")
            
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            self.logger.error(f"Error generating signature: {e}")
            return ''

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        if not self.rate_limit_enabled:
            return True
        
        now = time.time()
        # Remove requests older than the window
        self.request_times = [t for t in self.request_times if now - t < self.rate_limit_window]
        
        if len(self.request_times) >= self.rate_limit_requests:
            self.stats['rate_limited_requests'] += 1
            return False
        
        return True

    def _record_request(self):
        """Record a request for rate limiting."""
        if self.rate_limit_enabled:
            self.request_times.append(time.time())
        self.stats['total_requests'] += 1
        self.stats['last_request_time'] = datetime.now()

    async def _make_request(self, method: str, url: str, params: Dict[str, Any] = None, 
                           headers: Dict[str, str] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make HTTP request with error handling and rate limiting.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            headers: Request headers
            data: Request data

        Returns:
            Dict containing response data or error information
        """
        if not AIOHTTP_AVAILABLE:
            raise BinanceDependencyError("aiohttp is required for HTTP requests")
        
        if not self._check_rate_limit():
            raise BinanceAPIError("Rate limit exceeded")
        
        self._record_request()
        
        try:
            if not self.session:
                # Create session if not exists
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                data=data
            ) as response:
                
                if response.status == 200:
                    self.stats['successful_requests'] += 1
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    self.stats['rate_limited_requests'] += 1
                    error_text = await response.text()
                    raise BinanceAPIError(f"Rate limited: {error_text}")
                else:
                    self.stats['failed_requests'] += 1
                    error_text = await response.text()
                    raise BinanceAPIError(f"HTTP {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            self.stats['failed_requests'] += 1
            raise BinanceAPIError(f"Network error: {e}")
        except Exception as e:
            self.stats['failed_requests'] += 1
            raise BinanceAPIError(f"Request error: {e}")

    async def _initialize_ccxt(self) -> bool:
        """Initialize CCXT exchange as fallback."""
        if not CCXT_AVAILABLE:
            self.logger.warning("CCXT not available for fallback")
            return False
        
        try:
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt_async, 'binance')
            self.ccxt_exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.use_testnet,
                'timeout': self.timeout * 1000,  # CCXT uses milliseconds
                'enableRateLimit': True,
                'rateLimit': 100,  # 100ms between requests
            })
            
            # Test connection
            await self.ccxt_exchange.load_markets()
            self.logger.info("✅ CCXT fallback initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CCXT fallback: {e}")
            return False

    async def _ccxt_fallback_request(self, method: str, *args, **kwargs) -> Any:
        """Make request using CCXT as fallback."""
        if not self.ccxt_exchange:
            if not await self._initialize_ccxt():
                raise BinanceAPIError("CCXT fallback not available")
        
        try:
            self.stats['ccxt_fallback_used'] += 1
            self.logger.info(f"Using CCXT fallback for {method}")
            
            if hasattr(self.ccxt_exchange, method):
                result = await getattr(self.ccxt_exchange, method)(*args, **kwargs)
                self.stats['successful_requests'] += 1
                return result
            else:
                raise BinanceAPIError(f"Method {method} not available in CCXT")
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            raise BinanceAPIError(f"CCXT fallback failed: {e}")

    @handles_errors(context="Binance exchange initialization")
    async def initialize(self) -> bool:
        """
        Initialize Binance exchange connection.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("🔌 Initializing Binance Exchange...")
        
        try:
            if not AIOHTTP_AVAILABLE and not CCXT_AVAILABLE:
                self.logger.error("❌ Neither aiohttp nor CCXT available - cannot initialize connection")
                return False
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("❌ Invalid configuration")
                return False
            
            # Try primary API first
            if AIOHTTP_AVAILABLE:
                try:
                    server_time = await self._get_server_time()
                    if server_time:
                        self.is_connected = True
                        self.primary_api_failed = False
                        self.logger.info(f"✅ Connected to Binance API (Server time: {server_time})")
                        return True
                except Exception as e:
                    self.logger.warning(f"Primary API failed: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback and CCXT_AVAILABLE:
                try:
                    if await self._initialize_ccxt():
                        self.is_connected = True
                        self.logger.info("✅ Connected using CCXT fallback")
                        return True
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed: {e}")
            
            self.logger.error("❌ Failed to connect to Binance API")
            return False
                
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate exchange configuration."""
        if self.timeout <= 0:
            self.logger.error("Invalid timeout")
            return False
        if self.max_retries < 0:
            self.logger.error("Invalid max retries")
            return False
        if not self.use_testnet and (not self.api_key or not self.api_secret):
            self.logger.warning("API credentials required for live trading")
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _get_server_time(self) -> Optional[int]:
        """Get server time from Binance."""
        try:
            url = f'{self._get_base_url()}/api/v3/time'
            response = await self._make_request('GET', url)
            return response.get('serverTime')
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return None

    @with_error_recovery(service_name="binance_klines")
    @retry(stop=stop_after_attempt(3))
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 500) -> Optional[List[List[Any]]]:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            limit: Number of klines to retrieve

        Returns:
            Optional[List[List[Any]]]: Kline data or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {'symbol': symbol, 'interval': interval, 'limit': min(limit, 1000)}
                    url = f'{self._get_base_url()}/api/v3/klines'
                    
                    response = await self._make_request('GET', url, params=params)
                    self.logger.info(f'Klines retrieved: {len(response)} records')
                    return response
                except Exception as e:
                    self.logger.warning(f"Primary API failed for klines: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    result = await self._ccxt_fallback_request('fetch_ohlcv', symbol, interval, limit=limit)
                    self.logger.info(f'Klines retrieved via CCXT: {len(result)} records')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for klines: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting klines: {e}")
            return None

    @with_error_recovery(service_name="binance_ticker")
    @retry(stop=stop_after_attempt(3))
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker information.

        Args:
            symbol: Trading symbol

        Returns:
            Optional[Dict[str, Any]]: Ticker information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {'symbol': symbol}
                    url = f'{self._get_base_url()}/api/v3/ticker/24hr'
                    
                    response = await self._make_request('GET', url, params=params)
                    self.logger.info(f'Ticker retrieved: {symbol}')
                    return response
                except Exception as e:
                    self.logger.warning(f"Primary API failed for ticker: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    result = await self._ccxt_fallback_request('fetch_ticker', symbol)
                    self.logger.info(f'Ticker retrieved via CCXT: {symbol}')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for ticker: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            return None

    @with_error_recovery(service_name="binance_orderbook")
    @retry(stop=stop_after_attempt(3))
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get order book.

        Args:
            symbol: Trading symbol
            limit: Number of orders to retrieve

        Returns:
            Optional[Dict[str, Any]]: Order book or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {'symbol': symbol, 'limit': min(limit, 5000)}
                    url = f'{self._get_base_url()}/api/v3/depth'
                    
                    response = await self._make_request('GET', url, params=params)
                    self.logger.info(f'Order book retrieved: {symbol}')
                    return response
                except Exception as e:
                    self.logger.warning(f"Primary API failed for order book: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    result = await self._ccxt_fallback_request('fetch_order_book', symbol, limit)
                    self.logger.info(f'Order book retrieved via CCXT: {symbol}')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for order book: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            return None

    @with_error_recovery(service_name="binance_aggtrades")
    @retry(stop=stop_after_attempt(3))
    async def get_aggregate_trades(self, symbol: str, start_time_ms: int, end_time_ms: int) -> Optional[List[Dict[str, Any]]]:
        """
        Get aggregate trades for a symbol within a time range.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            Optional[List[Dict[str, Any]]]: Aggregate trades or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {
                        'symbol': symbol,
                        'startTime': start_time_ms,
                        'endTime': end_time_ms,
                        'limit': 1000
                    }
                    url = f'{self._get_base_url()}/api/v3/aggTrades'
                    
                    response = await self._make_request('GET', url, params=params)
                    self.logger.info(f'Aggregate trades retrieved: {len(response)} records')
                    return response
                except Exception as e:
                    self.logger.warning(f"Primary API failed for aggtrades: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    # CCXT doesn't have direct aggtrades, use trades instead
                    result = await self._ccxt_fallback_request('fetch_trades', symbol, limit=1000)
                    self.logger.info(f'Aggregate trades retrieved via CCXT: {len(result)} records')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for aggtrades: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting aggregate trades: {e}")
            return None

    @with_error_recovery(service_name="binance_futures")
    @retry(stop=stop_after_attempt(3))
    async def futures_funding_rate(self, symbol: str, start_time_ms: int, end_time_ms: int) -> Optional[List[Dict[str, Any]]]:
        """
        DEPRECATED: Get futures funding rates for a symbol within a time range.

        NOTE: funding_rate support has been removed from the data processing pipeline.
        This method is kept for backward compatibility but will return None.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            Optional[List[Dict[str, Any]]]: Always returns None (funding_rate support removed)
        """
        import warnings
        warnings.warn("futures_funding_rate method is deprecated. funding_rate support has been removed from the data processing pipeline.", DeprecationWarning, stacklevel=2)
        return None

    @with_error_recovery(service_name="binance_account")
    @retry(stop=stop_after_attempt(3))
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information.

        Returns:
            Optional[Dict[str, Any]]: Account information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for account info")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {'timestamp': int(time.time() * 1000)}
                    signature = self._generate_signature(params)
                    params['signature'] = signature
                    
                    url = f'{self._get_base_url()}/api/v3/account'
                    headers = {'X-MBX-APIKEY': self.api_key}
                    
                    response = await self._make_request('GET', url, params=params, headers=headers)
                    self.logger.info('Account information retrieved')
                    return response
                except Exception as e:
                    self.logger.warning(f"Primary API failed for account info: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    result = await self._ccxt_fallback_request('fetch_balance')
                    self.logger.info('Account information retrieved via CCXT')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for account info: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    @with_error_recovery(service_name="binance_position_risk")
    @retry(stop=stop_after_attempt(3))
    async def get_position_risk(self, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get position risk information.

        Args:
            symbol: Optional symbol filter

        Returns:
            Optional[List[Dict[str, Any]]]: Position risk information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None
            
            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for position risk")
                return None
            
            # Try primary API first
            if not self.primary_api_failed and AIOHTTP_AVAILABLE:
                try:
                    params = {'timestamp': int(time.time() * 1000), 'recvWindow': 5000}
                    if symbol:
                        params['symbol'] = symbol
                    
                    signature = self._generate_signature(params)
                    params['signature'] = signature
                    
                    url = f'{self._get_futures_base_url()}/fapi/v2/positionRisk'
                    headers = {'X-MBX-APIKEY': self.api_key}
                    
                    response = await self._make_request('GET', url, params=params, headers=headers)
                    self.logger.info('Position risk information retrieved')
                    return response if isinstance(response, list) else [response]
                except Exception as e:
                    self.logger.warning(f"Primary API failed for position risk: {e}")
                    self.primary_api_failed = True
            
            # Try CCXT fallback
            if self.use_ccxt_fallback:
                try:
                    result = await self._ccxt_fallback_request('fetch_positions', symbol)
                    self.logger.info('Position risk information retrieved via CCXT')
                    return result
                except Exception as e:
                    self.logger.error(f"CCXT fallback failed for position risk: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting position risk: {e}")
            return None

    def get_exchange_status(self) -> Dict[str, Any]:
        """Get exchange status information."""
        return {
            'is_connected': self.is_connected,
            'use_testnet': self.use_testnet,
            'base_url': self._get_base_url(),
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'api_key_configured': bool(self.api_key),
            'api_secret_configured': bool(self.api_secret),
            'primary_api_failed': self.primary_api_failed,
            'use_ccxt_fallback': self.use_ccxt_fallback,
            'dependencies': {
                'aiohttp': AIOHTTP_AVAILABLE,
                'ccxt': CCXT_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'numpy': NUMPY_AVAILABLE,
                'tenacity': TENACITY_AVAILABLE
            },
            'stats': self.stats
        }

    async def stop(self) -> None:
        """Stop the Binance exchange."""
        self.logger.info('🛑 Stopping Binance Exchange...')
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.ccxt_exchange:
                await self.ccxt_exchange.close()
                self.ccxt_exchange = None
            
            self.is_connected = False
            self.logger.info('✅ Binance Exchange stopped successfully')
        except Exception as e:
            self.logger.error(f'Error stopping Binance exchange: {e}')

    @staticmethod
    def get_exchange(exchange_name: str) -> 'BinanceExchange':
        """
        Factory method to create exchange instance.

        Args:
            exchange_name: Name of the exchange

        Returns:
            BinanceExchange instance
        """
        if exchange_name.lower() == 'binance':
            return BinanceExchange({})
        else:
            raise ValueError(f'Unsupported exchange: {exchange_name}')

# Backward compatibility
binance_exchange: BinanceExchange | None = None