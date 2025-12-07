"""
Binance Klines Data Adapter

This module provides a unified adapter for Binance klines data that ensures
complete equivalency with other exchanges and full compatibility with src/utils/data/.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import ssl

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from exchanges.shared import KlinesDataProcessingPipeline
from exchanges.shared.enhanced_unified_exchange_interface import (
    EnhancedUnifiedExchangeAdapter, ExchangeType, get_enhanced_standardized_klines
)
from exchanges.shared.unified_exchange_standardizer import (
    UnifiedExchangeStandardizer, DataQualityLevel
)

# Optional import to avoid circular dependencies
try:
    from exchanges.binance import BinanceExchange
    BINANCE_EXCHANGE_AVAILABLE = True
except ImportError:
    BINANCE_EXCHANGE_AVAILABLE = False


class BinanceKlinesAdapter:
    """Unified adapter for Binance klines data that ensures complete equivalency with other exchanges."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, data_dir: str = "historical_data"):
        """Initialize the Binance klines adapter.
        
        Args:
            api_key: Binance API key (optional for public data)
            secret_key: Binance secret key (optional for public data)
            data_dir: Directory for data storage
        """
        self.exchange = "binance"
        self.data_dir = data_dir
        
        # Initialize Binance exchange for API calls (if available).
        #
        # In this environment we primarily rely on the public REST klines
        # endpoint and treat the full trading stack as optional. Some
        # versions of BinanceExchange require additional constructor
        # arguments (e.g. trade_symbol), so we must guard against
        # signature mismatches and fall back cleanly to "public only" mode.
        self.binance_exchange = None
        if BINANCE_EXCHANGE_AVAILABLE:
            try:
                # Older versions accept (api_key, api_secret) only; if the
                # signature is different we ignore the failure and rely on
                # the public REST fallback instead of raising.
                self.binance_exchange = BinanceExchange(api_key, secret_key)  # type: ignore[arg-type]
            except TypeError:
                # Constructor signature mismatch – disable unified adapter
                self.binance_exchange = None
            except Exception:
                # Any other initialization error: treat the exchange as
                # unavailable and continue with public REST only.
                self.binance_exchange = None
        
        # Initialize shared processing pipeline
        self.processing_pipeline = KlinesDataProcessingPipeline(self.exchange, data_dir)
        
        # Initialize enhanced unified adapter for standardized data access
        if self.binance_exchange:
            self.unified_adapter = EnhancedUnifiedExchangeAdapter(
                self.binance_exchange, 
                ExchangeType.BINANCE,
                DataQualityLevel.STANDARD
            )
        else:
            self.unified_adapter = None
    
    async def get_klines_data(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get standardized klines data from Binance.
        
        Tries unified_adapter first (if available), otherwise falls back to
        public REST klines which do not require API keys.
        """
        # 1) Try unified adapter if available
        if self.unified_adapter:
            try:
                standardized_data = await self.unified_adapter.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                if standardized_data is not None and not standardized_data.empty:
                    return standardized_data
            except Exception as e:
                print(f"⚠️ Unified adapter klines failed, will try public REST: {e}")

        # 2) Public REST fallback (no API key required)
        try:
            raw_klines = await self._fetch_public_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            if not raw_klines:
                return pd.DataFrame()
            formatted = self._format_klines_data(raw_klines, symbol, interval)
            return formatted
        except Exception as e:
            print(f"❌ Error getting Binance klines data (public): {e}")
            return pd.DataFrame()

    async def _fetch_public_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch klines via Binance public REST API (no auth required)."""
        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": self._convert_interval(interval),
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        print(
            f"[BINANCE_PUBLIC] Requesting klines: symbol={symbol}, interval={interval}, "
            f"start_time={start_time}, end_time={end_time}, limit={limit}, params={params}"
        )

        async def _fetch_with_aiohttp():
            """Fetch klines using aiohttp with SSL verification, with
            a one-time fallback that disables verification if we hit
            CERTIFICATE_VERIFY_FAILED. This is intended for environments
            with intercepting proxies/self-signed CAs.
            """

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(base_url, params=params, timeout=30) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        print(f"[BINANCE_PUBLIC] aiohttp verified fetch ok, raw_len={len(data)}")
                        return data
                except Exception as e:
                    msg = str(e)
                    if "CERTIFICATE_VERIFY_FAILED" not in msg:
                        print(f"[BINANCE_PUBLIC] aiohttp fetch error (non-SSL): {e}")
                        raise
                    print("[BINANCE_PUBLIC] aiohttp CERTIFICATE_VERIFY_FAILED, retrying with ssl disabled")

            # Retry once with SSL verification disabled
            insecure_context = ssl.create_default_context()
            insecure_context.check_hostname = False
            insecure_context.verify_mode = ssl.CERT_NONE

            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params, timeout=30, ssl=insecure_context) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    print(f"[BINANCE_PUBLIC] aiohttp insecure fetch ok, raw_len={len(data)}")
                    return data

        def _fetch_with_urllib():
            """Fetch klines using urllib with a similar SSL fallback."""
            from urllib.parse import urlencode
            from urllib.request import urlopen
            import json

            url = f"{base_url}?{urlencode(params)}"

            print(f"[BINANCE_PUBLIC] urllib requesting URL: {url}")

            try:
                with urlopen(url, timeout=30) as resp:
                    payload = resp.read().decode()
                    data = json.loads(payload)
                    print(f"[BINANCE_PUBLIC] urllib verified fetch ok, raw_len={len(data)}")
                    return data
            except Exception as e:
                msg = str(e)
                if "CERTIFICATE_VERIFY_FAILED" not in msg:
                    print(f"[BINANCE_PUBLIC] urllib fetch error (non-SSL): {e}")
                    raise
                print("[BINANCE_PUBLIC] urllib CERTIFICATE_VERIFY_FAILED, retrying with ssl disabled")

            # Retry once with verification disabled
            insecure_context = ssl._create_unverified_context()
            with urlopen(url, timeout=30, context=insecure_context) as resp:
                payload = resp.read().decode()
                data = json.loads(payload)
                print(f"[BINANCE_PUBLIC] urllib insecure fetch ok, raw_len={len(data)}")
                return data

        data = await _fetch_with_aiohttp() if AIOHTTP_AVAILABLE else await asyncio.to_thread(_fetch_with_urllib)

        # Binance returns list of lists; convert to list of dicts for formatting
        klines: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, (list, tuple)) or len(entry) < 6:
                continue
            klines.append({
                "openTime": entry[0],
                "open": entry[1],
                "high": entry[2],
                "low": entry[3],
                "close": entry[4],
                "volume": entry[5],
                "closeTime": entry[6] if len(entry) > 6 else None,
                "quoteVolume": entry[7] if len(entry) > 7 else None,
                "trades": entry[8] if len(entry) > 8 else None,
                "takerBuyBase": entry[9] if len(entry) > 9 else None,
                "takerBuyQuote": entry[10] if len(entry) > 10 else None,
            })
        return klines
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Binance format.
        
        Args:
            interval: Standard interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Binance interval format
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return interval_map.get(interval, '1m')
    
    def _format_klines_data(self, data: List[Dict], symbol: str, interval: str) -> pd.DataFrame:
        """Format Binance klines data to standard format."""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        column_mapping = {
            'openTime': 'open_time',
            'closeTime': 'close_time', 
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'quoteVolume': 'quote_volume',
            'trades': 'trades',
            'takerBuyBase': 'taker_buy_base',
            'takerBuyQuote': 'taker_buy_quote'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('Int64')
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce').astype('Int64')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    async def download_and_process_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_data: bool = True
    ) -> pd.DataFrame:
        """Download and process klines data using the shared pipeline.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_time: Start time for data
            end_time: End time for data
            save_data: Whether to save processed data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Get raw data from API
            raw_data = await self.get_klines_data(symbol, interval, start_time, end_time)
            
            if raw_data.empty:
                return pd.DataFrame()
            
            # Process using shared pipeline
            processed_data = self.processing_pipeline.process_klines_data(
                raw_data, symbol, interval, save_data=save_data
            )
            
            return processed_data
            
        except Exception as e:
            print(f"❌ Error in download and process: {e}")
            return pd.DataFrame()
    
    def get_processed_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get previously processed data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Processed DataFrame or None
        """
        return self.processing_pipeline.get_processed_data(symbol, interval, start_date, end_date)
    
    def validate_data_quality(self, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Validate data quality using shared pipeline.
        
        Args:
            df: DataFrame to validate
            context: Validation context
            
        Returns:
            Validation results
        """
        return self.processing_pipeline.validate_data_quality(df, context)


# Convenience function for easy usage
def create_binance_klines_adapter(api_key: str = None, secret_key: str = None, data_dir: str = "historical_data") -> BinanceKlinesAdapter:
    """Create a Binance klines adapter instance.
    
    Args:
        api_key: Binance API key
        secret_key: Binance secret key
        data_dir: Data directory
        
    Returns:
        BinanceKlinesAdapter instance
    """
    return BinanceKlinesAdapter(api_key, secret_key, data_dir)