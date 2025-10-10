"""
OKX Klines Data Adapter

This module provides a minimal adapter for OKX klines data that handles
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
    from exchanges.okx import OkxExchange
    OKX_EXCHANGE_AVAILABLE = True
except ImportError:
    OKX_EXCHANGE_AVAILABLE = False


class OkxKlinesAdapter:
    """Minimal adapter for OKX klines data API calls and formatting."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, data_dir: str = "historical_data"):
        """Initialize the OKX klines adapter.
        
        Args:
            api_key: OKX API key (optional for public data)
            secret_key: OKX secret key (optional for public data)
            data_dir: Directory for data storage
        """
        self.exchange = "okx"
        self.data_dir = data_dir
        
        # Initialize OKX exchange for API calls (if available)
        if OKX_EXCHANGE_AVAILABLE:
            self.okx_exchange = OkxExchange(api_key, secret_key)
        else:
            self.okx_exchange = None
        
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
        """Get klines data from OKX API.
        
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
            if not self.okx_exchange:
                print("❌ OKX exchange not available - cannot fetch data")
                return pd.DataFrame()
            
            # Convert interval to OKX format
            okx_interval = self._convert_interval(interval)
            
            # Get data from OKX API
            klines_data = await self.okx_exchange.get_klines(
                symbol=symbol,
                interval=okx_interval,
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
            print(f"❌ Error getting OKX klines data: {e}")
            return pd.DataFrame()
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to OKX format.
        
        Args:
            interval: Standard interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            OKX interval format
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        return interval_map.get(interval, '1m')
    
    def _format_klines_data(self, data: List[Dict], symbol: str, interval: str) -> pd.DataFrame:
        """Format OKX klines data to standard format.
        
        Args:
            data: Raw klines data from OKX API
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Formatted DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns to standard format
        column_mapping = {
            'ts': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'volCcy': 'quote_volume',
            'volCcyQuote': 'quote_volume',
            'confirm': 'trades'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Convert timestamp column to int
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')
        
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
def create_okx_klines_adapter(api_key: str = None, secret_key: str = None, data_dir: str = "historical_data") -> OkxKlinesAdapter:
    """Create an OKX klines adapter instance.
    
    Args:
        api_key: OKX API key
        secret_key: OKX secret key
        data_dir: Data directory
        
    Returns:
        OkxKlinesAdapter instance
    """
    return OkxKlinesAdapter(api_key, secret_key, data_dir)