#!/usr/bin/env python3
"""
Unified Data Downloader

This module provides centralized download functionality for all data types:
- Klines data
- Aggtrades data  
- Futures data

Consolidates functionality from multiple redundant downloaders into a single,
optimized implementation.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
from src.utils.validation import validate_data_quality
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("UnifiedDataDownloader")

class UnifiedDataDownloader:
    """Unified downloader for all data types with comprehensive error handling and validation."""
    
    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('UnifiedDataDownloader')
        
        # Initialize standardized parquet handler for compatibility
        try:
            from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
            self.parquet_handler = standardized_parquet_handler
        except ImportError:
            self.parquet_handler = None
            self.logger.warning("⚠️ Standardized parquet handler not available")
        
        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'start_time': None
        }
        
        # Initialize exchange instances cache
        self._exchange_instances = {}
        
    @handles_errors(fallback=False, context="download_klines")
    @log_all_calls
    async def download_klines(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download klines data for a symbol and exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            
        Returns:
            Tuple of (success, data, error_message)
        """
        self.logger.info(f"📥 Downloading klines data: {exchange}_{symbol}_{timeframe}")
        
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
                
            self.logger.info(f"📅 Download period: {start_date} to {end_date}")
            
            # Get exchange instance
            exchange_instance = await self._get_exchange_instance(exchange)
            if not exchange_instance:
                return False, [], f"Failed to initialize {exchange} exchange"
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Download data in batches
            all_data = []
            current_start = start_timestamp
            
            while current_start < end_timestamp:
                batch_data = await self._download_klines_batch(
                    exchange_instance, symbol, timeframe, current_start, end_timestamp, batch_size
                )
                
                if not batch_data:
                    break
                    
                all_data.extend(batch_data)
                
                # Update timestamp for next batch
                if batch_data:
                    current_start = batch_data[-1]['timestamp'] + 1
                else:
                    break
                    
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Update statistics
            self.download_stats['total_downloads'] += 1
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_rows'] += len(all_data)
            
            self.logger.info(f"✅ Downloaded {len(all_data)} klines records")
            return True, all_data, None
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading klines: {e}")
            self.download_stats['failed_downloads'] += 1
            return False, [], str(e)
    
    @handles_errors(fallback=False, context="download_aggtrades")
    @log_all_calls
    async def download_aggtrades(
        self, 
        symbol: str, 
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download aggtrades data for a symbol and exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            
        Returns:
            Tuple of (success, data, error_message)
        """
        self.logger.info(f"📥 Downloading aggtrades data: {exchange}_{symbol}")
        
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)  # Shorter default for aggtrades
            if end_date is None:
                end_date = datetime.now()
                
            self.logger.info(f"📅 Download period: {start_date} to {end_date}")
            
            # Get exchange instance
            exchange_instance = await self._get_exchange_instance(exchange)
            if not exchange_instance:
                return False, [], f"Failed to initialize {exchange} exchange"
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Download data in batches
            all_data = []
            current_start = start_timestamp
            
            while current_start < end_timestamp:
                batch_data = await self._download_aggtrades_batch(
                    exchange_instance, symbol, current_start, end_timestamp, batch_size
                )
                
                if not batch_data:
                    break
                    
                all_data.extend(batch_data)
                
                # Update timestamp for next batch
                if batch_data:
                    current_start = batch_data[-1]['timestamp'] + 1
                else:
                    break
                    
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Update statistics
            self.download_stats['total_downloads'] += 1
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_rows'] += len(all_data)
            
            self.logger.info(f"✅ Downloaded {len(all_data)} aggtrades records")
            return True, all_data, None
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading aggtrades: {e}")
            self.download_stats['failed_downloads'] += 1
            return False, [], str(e)
    
    @handles_errors(fallback=False, context="download_futures")
    @log_all_calls
    async def download_futures(
        self, 
        symbol: str, 
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download futures data for a symbol and exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            
        Returns:
            Tuple of (success, data, error_message)
        """
        self.logger.info(f"📥 Downloading futures data: {exchange}_{symbol}")
        
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=90)  # Longer default for futures
            if end_date is None:
                end_date = datetime.now()
                
            self.logger.info(f"📅 Download period: {start_date} to {end_date}")
            
            # Get exchange instance
            exchange_instance = await self._get_exchange_instance(exchange)
            if not exchange_instance:
                return False, [], f"Failed to initialize {exchange} exchange"
            
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Download data in batches
            all_data = []
            current_start = start_timestamp
            
            while current_start < end_timestamp:
                batch_data = await self._download_futures_batch(
                    exchange_instance, symbol, current_start, end_timestamp, batch_size
                )
                
                if not batch_data:
                    break
                    
                all_data.extend(batch_data)
                
                # Update timestamp for next batch
                if batch_data:
                    current_start = batch_data[-1]['timestamp'] + 1
                else:
                    break
                    
                # Rate limiting
                await asyncio.sleep(0.1)
            
            # Update statistics
            self.download_stats['total_downloads'] += 1
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_rows'] += len(all_data)
            
            self.logger.info(f"✅ Downloaded {len(all_data)} futures records")
            return True, all_data, None
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading futures: {e}")
            self.download_stats['failed_downloads'] += 1
            return False, [], str(e)
    
    @handles_errors(fallback=None, context="get_exchange_instance")
    async def _get_exchange_instance(self, exchange: str):
        """Get or create exchange instance."""
        if exchange.upper() not in self._exchange_instances:
            try:
                # Import exchange dynamically
                exchange_module = f"src.exchanges.{exchange.lower()}"
                exchange_class = f"{exchange.title()}Exchange"
                
                module = __import__(exchange_module, fromlist=[exchange_class])
                exchange_class_obj = getattr(module, exchange_class)
                
                self._exchange_instances[exchange.upper()] = exchange_class_obj()
                self.logger.info(f"✅ Initialized {exchange} exchange")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {exchange} exchange: {e}")
                return None
                
        return self._exchange_instances[exchange.upper()]
    
    @handles_errors(fallback=[], context="download_klines_batch")
    async def _download_klines_batch(
        self, 
        exchange_instance, 
        symbol: str, 
        timeframe: str, 
        start_timestamp: int, 
        end_timestamp: int, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Download a batch of klines data."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            # Download from exchange
            raw_data = await exchange_instance.fetch_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_dt,
                end_time=end_dt,
                limit=batch_size
            )
            
            # Convert to standardized format
            standardized_data = []
            for item in raw_data:
                standardized_data.append({
                    'timestamp': item.get('timestamp', item.get('open_time', 0)),
                    'open': float(item.get('open', 0)),
                    'high': float(item.get('high', 0)),
                    'low': float(item.get('low', 0)),
                    'close': float(item.get('close', 0)),
                    'volume': float(item.get('volume', 0)),
                    'symbol': symbol,
                    'exchange': exchange_instance.name.upper(),
                    'timeframe': timeframe
                })
            
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading klines batch: {e}")
            return []
    
    @handles_errors(fallback=[], context="download_aggtrades_batch")
    async def _download_aggtrades_batch(
        self, 
        exchange_instance, 
        symbol: str, 
        start_timestamp: int, 
        end_timestamp: int, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Download a batch of aggtrades data."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            # Download from exchange
            raw_data = await exchange_instance.fetch_aggtrades(
                symbol=symbol,
                start_time=start_dt,
                end_time=end_dt,
                limit=batch_size
            )
            
            # Convert to standardized format
            standardized_data = []
            for item in raw_data:
                standardized_data.append({
                    'timestamp': item.get('timestamp', item.get('T', 0)),
                    'price': float(item.get('price', item.get('p', 0))),
                    'quantity': float(item.get('quantity', item.get('q', 0))),
                    'is_buyer_maker': item.get('is_buyer_maker', item.get('m', False)),
                    'trade_id': item.get('trade_id', item.get('a', 0)),
                    'symbol': symbol,
                    'exchange': exchange_instance.name.upper()
                })
            
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading aggtrades batch: {e}")
            return []
    
    @handles_errors(fallback=[], context="download_futures_batch")
    async def _download_futures_batch(
        self, 
        exchange_instance, 
        symbol: str, 
        start_timestamp: int, 
        end_timestamp: int, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Download a batch of futures data."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            # Download from exchange
            raw_data = await exchange_instance.fetch_futures(
                symbol=symbol,
                start_time=start_dt,
                end_time=end_dt,
                limit=batch_size
            )
            
            # Convert to standardized format
            standardized_data = []
            for item in raw_data:
                standardized_data.append({
                    'timestamp': item.get('timestamp', item.get('fundingTime', 0)),
                    'funding_rate': float(item.get('funding_rate', item.get('fundingRate', 0))),
                    'symbol': symbol,
                    'exchange': exchange_instance.name.upper()
                })
            
            return standardized_data
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading futures batch: {e}")
            return []
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        return {
            **self.download_stats,
            'success_rate': (
                self.download_stats['successful_downloads'] / 
                max(self.download_stats['total_downloads'], 1) * 100
            )
        }
    
    def reset_stats(self):
        """Reset download statistics."""
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'start_time': None
        }

# Convenience functions for backward compatibility
@handles_errors(fallback=False)
async def download_klines_data(symbol: str, exchange: str, timeframe: str = "1m", **kwargs) -> bool:
    """Convenience function for downloading klines data."""
    downloader = UnifiedDataDownloader()
    success, data, error = await downloader.download_klines(symbol, exchange, timeframe, **kwargs)
    return success

@handles_errors(fallback=False)
async def download_aggtrades_data(symbol: str, exchange: str, **kwargs) -> bool:
    """Convenience function for downloading aggtrades data."""
    downloader = UnifiedDataDownloader()
    success, data, error = await downloader.download_aggtrades(symbol, exchange, **kwargs)
    return success

@handles_errors(fallback=False)
async def download_futures_data(symbol: str, exchange: str, **kwargs) -> bool:
    """Convenience function for downloading futures data."""
    downloader = UnifiedDataDownloader()
    success, data, error = await downloader.download_futures(symbol, exchange, **kwargs)
    return success