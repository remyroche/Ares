"""
BingX Klines Data Adapter

This module provides a minimal adapter for BingX klines data that handles
API-specific formatting and compatibility with the ExchangeInterface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from exchanges.shared import KlinesDataProcessingPipeline

# Optional import to avoid circular dependencies
try:
    from exchanges.bingx import BingXExchange
    BINGX_EXCHANGE_AVAILABLE = True
except ImportError:
    BINGX_EXCHANGE_AVAILABLE = False


class BingXKlinesAdapter:
    """Minimal adapter for BingX klines data API calls and formatting."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, data_dir: str = "historical_data"):
        """Initialize the BingX klines adapter.
        
        Args:
            api_key: BingX API key (optional for public data)
            secret_key: BingX secret key (optional for public data)
            data_dir: Directory for data storage
        """
        self.exchange = "bingx"
        self.data_dir = data_dir
        
        # Initialize BingX exchange for API calls (if available)
        if BINGX_EXCHANGE_AVAILABLE:
            self.bingx_exchange = BingXExchange(api_key, secret_key)
        else:
            self.bingx_exchange = None
        
        # Initialize shared processing pipeline
        self.processing_pipeline = KlinesDataProcessingPipeline(self.exchange, data_dir)
    
    async def get_klines_data(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get klines data from BingX API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of records
            
        Returns:
            DataFrame with klines data
        """
        try:
            if not self.bingx_exchange:
                print("❌ BingX exchange not available - cannot fetch data")
                return pd.DataFrame()
            
            # Convert interval to BingX format
            bingx_interval = self._convert_interval(interval)
            
            # Get data from BingX API
            klines_data = await self.bingx_exchange.get_klines(
                symbol=symbol,
                interval=bingx_interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if not klines_data or klines_data.empty:
                return pd.DataFrame()
            
            # Convert to standard format
            standardized_data = self._format_klines_data(klines_data, symbol, interval)
            
            return standardized_data
            
        except Exception as e:
            print(f"❌ Error getting BingX klines data: {e}")
            return pd.DataFrame()
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to BingX format.
        
        Args:
            interval: Standard interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            BingX interval format
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return interval_map.get(interval, '1m')
    
    def _format_klines_data(self, data: List[Dict], symbol: str, interval: str) -> pd.DataFrame:
        """Format BingX klines data to standard format.
        
        Args:
            data: Raw klines data from BingX API
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Formatted DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        # Extract the 'data' field from BingX response
        klines = data.get('data', []) if isinstance(data, dict) else data
        
        if not klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        
        # Rename columns to standard format
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
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure required columns exist
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Convert timestamp columns to int
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce').astype('Int64')
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce').astype('Int64')
        
        # Convert OHLCV to numeric
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
def create_bingx_klines_adapter(api_key: str = None, secret_key: str = None, data_dir: str = "historical_data") -> BingXKlinesAdapter:
    """Create a BingX klines adapter instance.
    
    Args:
        api_key: BingX API key
        secret_key: BingX secret key
        data_dir: Data directory
        
    Returns:
        BingXKlinesAdapter instance
    """
    return BingXKlinesAdapter(api_key, secret_key, data_dir)