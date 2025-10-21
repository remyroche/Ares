"""
Phemex Klines Data Adapter

This module provides a minimal adapter for Phemex klines data that handles
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
from src.utils.tprint import tprint_data_preview

# Optional import to avoid circular dependencies
try:
    from exchanges.phemex import PhemexExchange
    PHEMEX_EXCHANGE_AVAILABLE = True
except ImportError:
    PHEMEX_EXCHANGE_AVAILABLE = False


class PhemexKlinesAdapter:
    """Minimal adapter for Phemex klines data API calls and formatting."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, data_dir: str = "historical_data"):
        """Initialize the Phemex klines adapter.
        
        Args:
            api_key: Phemex API key (optional for public data)
            secret_key: Phemex secret key (optional for public data)
            data_dir: Directory for data storage
        """
        self.exchange = "phemex"
        self.data_dir = data_dir
        
        # Initialize Phemex exchange for API calls (if available)
        if PHEMEX_EXCHANGE_AVAILABLE:
            self.phemex_exchange = PhemexExchange(api_key, secret_key)
        else:
            self.phemex_exchange = None
        
        # Initialize shared processing pipeline
        self.processing_pipeline = KlinesDataProcessingPipeline(self.exchange, data_dir)
    
    async def get_klines_data(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        enable_data_preview: bool = True
    ) -> pd.DataFrame:
        """Get klines data from Phemex API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of records
            enable_data_preview: Whether to show data preview using tprint_data_preview
            
        Returns:
            DataFrame with klines data
        """
        try:
            if not self.phemex_exchange:
                print("❌ Phemex exchange not available - cannot fetch data")
                return pd.DataFrame()
            
            # Convert interval to Phemex format
            phemex_interval = self._convert_interval(interval)
            
            # Get data from Phemex API
            klines_data = await self.phemex_exchange.get_klines(
                symbol=symbol,
                interval=phemex_interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if not klines_data or klines_data.empty:
                return pd.DataFrame()
            
            # Convert to standard format
            standardized_data = self._format_klines_data(klines_data, symbol, interval)
            
            # Show data preview if enabled
            if enable_data_preview and not standardized_data.empty:
                tprint_data_preview(
                    standardized_data, 
                    name=f"Phemex klines data for {symbol} ({interval})",
                    max_rows=5,
                    level="INFO"
                )
            
            return standardized_data
            
        except Exception as e:
            print(f"❌ Error getting Phemex klines data: {e}")
            return pd.DataFrame()
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Phemex format.
        
        Args:
            interval: Standard interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Phemex interval format
        """
        interval_map = {
            '1m': '60',
            '5m': '300', 
            '15m': '900',
            '30m': '1800',
            '1h': '3600',
            '4h': '14400',
            '1d': '86400'
        }
        return interval_map.get(interval, '60')
    
    def _format_klines_data(self, data: List[Dict], symbol: str, interval: str) -> pd.DataFrame:
        """Format Phemex klines data to standard format.
        
        Args:
            data: Raw klines data from Phemex API
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
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'turnover': 'quote_volume',
            'trades': 'trades'
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
        save_data: bool = True,
        enable_data_preview: bool = True
    ) -> pd.DataFrame:
        """Download and process klines data using the shared pipeline.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_time: Start time for data
            end_time: End time for data
            save_data: Whether to save processed data
            enable_data_preview: Whether to show data preview using tprint_data_preview
            
        Returns:
            Processed DataFrame
        """
        try:
            # Get raw data from API
            raw_data = await self.get_klines_data(symbol, interval, start_time, end_time, enable_data_preview=enable_data_preview)
            
            if raw_data.empty:
                return pd.DataFrame()
            
            # Process using shared pipeline
            processed_data = self.processing_pipeline.process_klines_data(
                raw_data, symbol, interval, save_data=save_data
            )
            
            # Show processed data preview if enabled
            if enable_data_preview and not processed_data.empty:
                tprint_data_preview(
                    processed_data, 
                    name=f"Processed Phemex klines data for {symbol} ({interval})",
                    max_rows=5,
                    level="INFO"
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
def create_phemex_klines_adapter(api_key: str = None, secret_key: str = None, data_dir: str = "historical_data") -> PhemexKlinesAdapter:
    """Create a Phemex klines adapter instance.
    
    Args:
        api_key: Phemex API key
        secret_key: Phemex secret key
        data_dir: Data directory
        
    Returns:
        PhemexKlinesAdapter instance
    """
    return PhemexKlinesAdapter(api_key, secret_key, data_dir)