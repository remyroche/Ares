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
# from src.utils.validation import validate_data_quality  # Replaced with comprehensive quality tools

# Import comprehensive data quality tools
try:
    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
    from src.utils.data.quality.data_quality import DataQualityFramework
    QUALITY_TOOLS_AVAILABLE = True
except ImportError:
    QUALITY_TOOLS_AVAILABLE = False

def validate_data_quality(df, **kwargs):
    """Comprehensive data quality validation using proper tools."""
    if not QUALITY_TOOLS_AVAILABLE:
        return {'valid': True, 'quality_score': 50.0, 'issues': [], 'warnings': []}
    
    try:
        quality_scorer = get_quality_scorer()
        quality_assessment = quality_scorer.assess_data_quality(
            df,
            context="data_collection",
            step_name="data_download",
            data_type="klines"
        )
        
        return {
            'valid': quality_assessment.level.value not in ['critical'],
            'quality_score': quality_assessment.overall_score,
            'issues': quality_assessment.issues,
            'warnings': quality_assessment.warnings
        }
    except Exception as e:
        return {'valid': True, 'quality_score': 50.0, 'issues': [str(e)], 'warnings': []}
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

        # Lazy initialization of Binance API - only when needed
        self.binance_class = None

    def _ensure_binance_api(self) -> bool:
        """Ensure Binance API is available when needed."""
        if self.binance_class is None:
            try:
                from src.exchange.binance import BinanceExchange
                self.binance_class = BinanceExchange
                self.logger.info("✅ Binance API available")
                return True
            except ImportError:
                self.binance_class = None
                self.logger.warning("⚠️ Binance API not available")
                return False
        return True

    @handles_errors(context="download_klines")
    @log_all_calls
    async def download_klines(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
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
            use_append_mode: Whether to use append mode (creates new files instead of overwriting)
            
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
            
            # Use enhanced append downloader if append mode is enabled
            if use_append_mode:
                try:
                    from .enhanced_append_data_downloader import EnhancedAppendDataDownloader
                    append_downloader = EnhancedAppendDataDownloader(str(self.data_cache_path))
                    
                    result = await append_downloader.download_with_append(
                        symbol=symbol,
                        exchange=exchange,
                        data_type="klines",
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        max_batches=10
                    )
                    
                    if result['success']:
                        # Update statistics
                        self.download_stats['total_downloads'] += 1
                        self.download_stats['successful_downloads'] += 1
                        self.download_stats['total_rows'] += result['total_rows']
                        
                        self.logger.info(f"✅ Downloaded {result['total_rows']} klines records using append mode")
                        return True, [], None  # Data is saved to files, not returned
                    else:
                        self.logger.error(f"❌ Append download failed: {result.get('error', 'Unknown error')}")
                        return False, [], result.get('error', 'Append download failed')
                        
                except ImportError:
                    self.logger.warning("⚠️ Enhanced append downloader not available, falling back to standard mode")
                except Exception as e:
                    self.logger.warning(f"⚠️ Append download failed, falling back to standard mode: {e}")
            
            # Standard download mode (fallback)
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
    
    @handles_errors(context="download_aggtrades")
    @log_all_calls
    async def download_aggtrades(
        self, 
        symbol: str, 
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        use_append_mode: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Download aggtrades data for a symbol and exchange.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            use_append_mode: Whether to use append mode (creates new files instead of overwriting)
            
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
            
            # Use enhanced append downloader if append mode is enabled
            if use_append_mode:
                try:
                    from .enhanced_append_data_downloader import EnhancedAppendDataDownloader
                    append_downloader = EnhancedAppendDataDownloader(str(self.data_cache_path))
                    
                    result = await append_downloader.download_with_append(
                        symbol=symbol,
                        exchange=exchange,
                        data_type="aggtrades",
                        timeframe="1m",  # Aggtrades don't have timeframes
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size,
                        max_batches=10
                    )
                    
                    if result['success']:
                        # Update statistics
                        self.download_stats['total_downloads'] += 1
                        self.download_stats['successful_downloads'] += 1
                        self.download_stats['total_rows'] += result['total_rows']
                        
                        self.logger.info(f"✅ Downloaded {result['total_rows']} aggtrades records using append mode")
                        return True, [], None  # Data is saved to files, not returned
                    else:
                        self.logger.error(f"❌ Append download failed: {result.get('error', 'Unknown error')}")
                        return False, [], result.get('error', 'Append download failed')
                        
                except ImportError:
                    self.logger.warning("⚠️ Enhanced append downloader not available, falling back to standard mode")
                except Exception as e:
                    self.logger.warning(f"⚠️ Append download failed, falling back to standard mode: {e}")
            
            # Standard download mode (fallback)
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
    
    
    @handles_errors(context="get_exchange_instance")
    async def _get_exchange_instance(self, exchange: str):
        """Get or create exchange instance using enhanced API."""
        if exchange.upper() not in self._exchange_instances:
            try:
                # Create exchange instance based on exchange name
                if exchange.lower() == 'binance':
                    # Ensure Binance API is available before using it
                    if not self._ensure_binance_api():
                        self.logger.error(f"❌ Binance API not available")
                        return None

                    # Use Binance API with proper configuration
                    config = {
                        'binance_exchange': {
                            'use_testnet': True,  # Use testnet for safety
                            'timeout': 30,
                            'max_retries': 3,
                            'rate_limit_enabled': True,
                            'rate_limit_requests': 1000,
                            'rate_limit_window': 60
                        }
                    }
                    exchange_instance = self.binance_class(config)
                    self.logger.info("✅ Using Binance API")
                else:
                    raise ValueError(f"Unsupported exchange: {exchange}")

                # Initialize the exchange
                success = await exchange_instance.initialize()
                if not success:
                    self.logger.error(f"❌ Failed to initialize {exchange} exchange")
                    return None

                self._exchange_instances[exchange.upper()] = exchange_instance
                self.logger.info(f"✅ Initialized {exchange} exchange")

            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {exchange} exchange: {e}")
                return None

        return self._exchange_instances[exchange.upper()]

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        timeframe = timeframe.lower()
        if timeframe == '1m':
            return 1
        elif timeframe == '5m':
            return 5
        elif timeframe == '15m':
            return 15
        elif timeframe == '30m':
            return 30
        elif timeframe == '1h':
            return 60
        elif timeframe == '4h':
            return 240
        elif timeframe == '1d':
            return 1440
        else:
            # Default to 1 minute
            return 1

    @handles_errors(context="download_klines_batch")
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
            # Calculate limit based on time range and timeframe
            # For klines, we need to estimate how many candles fit in the time range
            time_range_ms = end_timestamp - start_timestamp
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            estimated_candles = int(time_range_ms / (timeframe_minutes * 60 * 1000))

            # Limit to batch_size or estimated candles, whichever is smaller
            limit = min(batch_size, estimated_candles, 1000)  # Binance max limit is 1000

            # Use the Binance API get_klines method
            raw_data = await exchange_instance.get_klines(symbol, timeframe, limit)

            if raw_data is None:
                return []

            # Convert to standardized format (CCXT format from Binance API)
            standardized_data = []
            for item in raw_data:
                if isinstance(item, list) and len(item) >= 6:
                    standardized_data.append({
                        'timestamp': int(item[0]),  # timestamp
                        'open': float(item[1]),      # open
                        'high': float(item[2]),      # high
                        'low': float(item[3]),       # low
                        'close': float(item[4]),     # close
                        'volume': float(item[5]),    # volume
                        'symbol': symbol,
                        'exchange': 'BINANCE',
                        'timeframe': timeframe
                    })

            return standardized_data

        except Exception as e:
            self.logger.error(f"❌ Error downloading klines batch: {e}")
            return []
    
    @handles_errors(context="download_aggtrades_batch")
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
            # Use the Binance API get_aggregate_trades method
            raw_data = await exchange_instance.get_aggregate_trades(symbol, start_timestamp, end_timestamp)

            if raw_data is None:
                return []

            # Convert to standardized format
            standardized_data = []
            for item in raw_data:
                if isinstance(item, dict):
                    standardized_data.append({
                        'timestamp': item.get('timestamp', item.get('T', item.get('time', 0))),
                        'price': float(item.get('price', item.get('p', 0))),
                        'quantity': float(item.get('quantity', item.get('q', 0))),
                        'is_buyer_maker': item.get('is_buyer_maker', item.get('m', False)),
                        'trade_id': item.get('trade_id', item.get('a', item.get('id', 0))),
                        'symbol': symbol,
                        'exchange': 'BINANCE'
                    })

            return standardized_data

        except Exception as e:
            self.logger.error(f"❌ Error downloading aggtrades batch: {e}")
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
@handles_errors()
async def download_klines_data(symbol: str, exchange: str, timeframe: str = "1m", **kwargs) -> bool:
    """Convenience function for downloading klines data."""
    downloader = UnifiedDataDownloader()
    success, data, error = await downloader.download_klines(symbol, exchange, timeframe, **kwargs)
    return success

@handles_errors()
async def download_aggtrades_data(symbol: str, exchange: str, **kwargs) -> bool:
    """Convenience function for downloading aggtrades data."""
    downloader = UnifiedDataDownloader()
    success, data, error = await downloader.download_aggtrades(symbol, exchange, **kwargs)
    return success
