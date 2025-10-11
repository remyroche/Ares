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
        
        # Initialize Binance exchange for API calls (if available)
        if BINANCE_EXCHANGE_AVAILABLE:
            self.binance_exchange = BinanceExchange(api_key, secret_key)
        else:
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
        """Get standardized klines data from Binance API.
        
        This method ensures complete equivalency with other exchanges and
        full compatibility with src/utils/data/ utilities.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of records
            
        Returns:
            Standardized DataFrame compatible with src/utils/data/
        """
        try:
            if not self.unified_adapter:
                print("❌ Binance exchange not available - cannot fetch data")
                return pd.DataFrame()
            
            # Use unified adapter for standardized data
            standardized_data = await self.unified_adapter.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            return standardized_data
            
        except Exception as e:
            print(f"❌ Error getting Binance klines data: {e}")
            return pd.DataFrame()
    
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
        """Format Binance klines data to standard format.
        
        Args:
            data: Raw klines data from Binance API
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