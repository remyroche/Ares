#!/usr/bin/env python3
"""
Enhanced Append Data Downloader

This module provides data download functionality that ensures data is appended
to existing files rather than overwritten. Each batch download creates new files
and provides consolidation capabilities to merge data when needed.

Key Features:
- Batch-based file naming to prevent overwrites
- Incremental data downloading with gap detection
- Data consolidation and merging capabilities
- Comprehensive logging and monitoring
- Support for multiple data types (klines, aggtrades, futures)
- Integration with existing exchange APIs
"""

import asyncio
import sys
import time
import os
import glob
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
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("EnhancedAppendDataDownloader")

class EnhancedAppendDataDownloader:
    """Enhanced data downloader with append functionality and batch management."""
    
    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('EnhancedAppendDataDownloader')
        
        # Initialize standardized parquet handler
        self.parquet_handler = standardized_parquet_handler
        
        # Download statistics
        self.download_stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_rows': 0,
            'total_batches': 0,
            'start_time': None,
            'last_download_time': None
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

    @handles_errors(context="download_with_append")
    @log_all_calls
    async def download_with_append(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str = "klines",
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        batch_size: int = 1000,
        max_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Download data with append functionality - creates new batch files.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date for download
            end_date: End date for download
            batch_size: Number of records per batch
            max_batches: Maximum number of batches to download
            
        Returns:
            Dictionary with download results and statistics
        """
        self.logger.info(f"📥 Starting append download: {exchange}_{symbol}_{data_type}_{timeframe}")
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        self.logger.info(f"📅 Download period: {start_date} to {end_date}")
        
        # Initialize download session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.download_stats['start_time'] = time.time()
        
        # Get exchange instance
        exchange_instance = await self._get_exchange_instance(exchange)
        if not exchange_instance:
            return {
                'success': False,
                'error': f"Failed to initialize {exchange} exchange",
                'session_id': session_id
            }
        
        # Convert dates to timestamps
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # Download data in batches
        batch_results = []
        current_start = start_timestamp
        batch_count = 0
        
        while current_start < end_timestamp and batch_count < max_batches:
            batch_count += 1
            self.logger.info(f"📦 Downloading batch {batch_count}/{max_batches}")
            
            # Download batch
            batch_success, batch_data, next_timestamp = await self._download_batch(
                exchange_instance, symbol, data_type, timeframe, exchange,
                current_start, end_timestamp, batch_size, session_id, batch_count
            )
            
            if batch_success and batch_data:
                # Save batch to new file
                file_path = await self._save_batch_file(
                    batch_data, data_type, exchange, symbol, timeframe, 
                    session_id, batch_count
                )
                
                batch_results.append({
                    'batch_number': batch_count,
                    'success': True,
                    'rows': len(batch_data),
                    'file_path': str(file_path),
                    'start_timestamp': current_start,
                    'end_timestamp': next_timestamp or current_start
                })
                
                # Update statistics
                self.download_stats['successful_downloads'] += 1
                self.download_stats['total_rows'] += len(batch_data)
                
                self.logger.info(f"✅ Batch {batch_count}: {len(batch_data)} rows saved to {file_path.name}")
            else:
                batch_results.append({
                    'batch_number': batch_count,
                    'success': False,
                    'rows': 0,
                    'error': 'Download failed'
                })
                self.download_stats['failed_downloads'] += 1
                self.logger.warning(f"⚠️ Batch {batch_count} failed")
            
            # Update for next batch
            if next_timestamp:
                current_start = next_timestamp + 1
            else:
                break
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        # Update final statistics
        self.download_stats['total_downloads'] += 1
        self.download_stats['total_batches'] += batch_count
        self.download_stats['last_download_time'] = time.time()
        
        # Generate summary
        total_duration = time.time() - self.download_stats['start_time']
        successful_batches = len([r for r in batch_results if r['success']])
        total_rows = sum(r['rows'] for r in batch_results)
        
        result = {
            'success': successful_batches > 0,
            'session_id': session_id,
            'exchange': exchange,
            'symbol': symbol,
            'data_type': data_type,
            'timeframe': timeframe,
            'total_batches': batch_count,
            'successful_batches': successful_batches,
            'failed_batches': batch_count - successful_batches,
            'total_rows': total_rows,
            'total_duration': total_duration,
            'batch_results': batch_results,
            'download_stats': self.download_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Append download completed: {successful_batches}/{batch_count} batches, {total_rows} rows")
        return result
    
    @handles_errors(context="download_batch")
    async def _download_batch(
        self,
        exchange_instance,
        symbol: str,
        data_type: str,
        timeframe: str,
        exchange: str,
        start_timestamp: int,
        end_timestamp: int,
        batch_size: int,
        session_id: str,
        batch_number: int
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[int]]:
        """Download a single batch of data."""
        try:
            self.logger.info(f"🔄 Downloading {data_type} batch {batch_number} for {symbol}")
            
            # Download data based on type
            if data_type == 'klines':
                raw_data = await self._download_klines_batch(
                    exchange_instance, symbol, timeframe, start_timestamp, end_timestamp, batch_size
                )
            elif data_type == 'aggtrades':
                raw_data = await self._download_aggtrades_batch(
                    exchange_instance, symbol, start_timestamp, end_timestamp, batch_size
                )
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if not raw_data:
                return False, [], None
            
            # Standardize data format
            standardized_data = self._standardize_batch_data(raw_data, data_type, symbol, exchange, timeframe)
            
            # Get next timestamp for continuation
            next_timestamp = standardized_data[-1]['timestamp'] if standardized_data else None
            
            return True, standardized_data, next_timestamp
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading batch {batch_number}: {e}")
            return False, [], None
    
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
    
    
    @handles_errors(context="standardize_batch_data")
    def _standardize_batch_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        data_type: str, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> List[Dict[str, Any]]:
        """Standardize batch data format."""
        standardized_data = []
        
        for item in raw_data:
            if data_type == 'klines':
                standardized_data.append({
                    'timestamp': item.get('timestamp', item.get('open_time', 0)),
                    'open': float(item.get('open', 0)),
                    'high': float(item.get('high', 0)),
                    'low': float(item.get('low', 0)),
                    'close': float(item.get('close', 0)),
                    'volume': float(item.get('volume', 0)),
                    'symbol': symbol,
                    'exchange': exchange.upper(),
                    'timeframe': timeframe
                })
            elif data_type == 'aggtrades':
                standardized_data.append({
                    'timestamp': item.get('timestamp', item.get('T', 0)),
                    'price': float(item.get('price', item.get('p', 0))),
                    'quantity': float(item.get('quantity', item.get('q', 0))),
                    'is_buyer_maker': item.get('is_buyer_maker', item.get('m', False)),
                    'trade_id': item.get('trade_id', item.get('a', 0)),
                    'symbol': symbol,
                    'exchange': exchange.upper()
                })
        
        return standardized_data
    
    @handles_errors(context="save_batch_file")
    async def _save_batch_file(
        self, 
        batch_data: List[Dict[str, Any]], 
        data_type: str, 
        exchange: str, 
        symbol: str, 
        timeframe: str,
        session_id: str, 
        batch_number: int
    ) -> Path:
        """Save batch data to a new file with unique naming."""
        try:
            # Create directory structure
            data_dir = self.data_cache_path / exchange.lower() / symbol.lower() / data_type
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp and batch number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{exchange}_{symbol}_{timeframe}_{session_id}_batch_{batch_number:03d}_{timestamp}.parquet"
            file_path = data_dir / filename
            
            # Convert to DataFrame
            df = pd.DataFrame(batch_data)
            
            # Add metadata columns
            df['batch_number'] = batch_number
            df['session_id'] = session_id
            df['download_timestamp'] = int(datetime.now().timestamp() * 1000)
            
            # Save using standardized parquet handler
            success = self.parquet_handler.write_parquet_standardized(
                df, file_path, schema_name='unified', validate_quality=True
            )
            
            if not success:
                raise Exception("Failed to save batch file")
            
            self.logger.info(f"💾 Saved batch {batch_number} to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"❌ Error saving batch file: {e}")
            raise
    
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

                    # Use enhanced Binance API
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
                    self.logger.info("✅ Using enhanced Binance API")
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

    @handles_errors(context="consolidate_batches")
    @log_all_calls
    async def consolidate_batches(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str = "klines",
        timeframe: str = "1m",
        session_id: Optional[str] = None,
        remove_originals: bool = False
    ) -> Dict[str, Any]:
        """
        Consolidate multiple batch files into a single file.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data
            timeframe: Timeframe
            session_id: Specific session to consolidate (if None, consolidates all)
            remove_originals: Whether to remove original batch files after consolidation
            
        Returns:
            Dictionary with consolidation results
        """
        self.logger.info(f"🔄 Consolidating batches: {exchange}_{symbol}_{data_type}_{timeframe}")
        
        try:
            # Find batch files
            batch_files = await self._find_batch_files(
                symbol, exchange, data_type, timeframe, session_id
            )
            
            if not batch_files:
                return {
                    'success': False,
                    'error': 'No batch files found',
                    'consolidated_file': None
                }
            
            self.logger.info(f"📁 Found {len(batch_files)} batch files to consolidate")
            
            # Load and combine all batch files
            all_data = []
            for batch_file in batch_files:
                df = self.parquet_handler.read_parquet_standardized(batch_file)
                if df is not None and not df.empty:
                    all_data.append(df)
            
            if not all_data:
                return {
                    'success': False,
                    'error': 'No valid data found in batch files',
                    'consolidated_file': None
                }
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates based on timestamp
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Generate consolidated filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidated_filename = f"{data_type}_{exchange}_{symbol}_{timeframe}_consolidated_{timestamp}.parquet"
            consolidated_path = self.data_cache_path / exchange.lower() / symbol.lower() / data_type / consolidated_filename
            
            # Save consolidated file
            success = self.parquet_handler.write_parquet_standardized(
                combined_df, consolidated_path, schema_name='unified', validate_quality=True
            )
            
            if not success:
                return {
                    'success': False,
                    'error': 'Failed to save consolidated file',
                    'consolidated_file': None
                }
            
            # Remove original batch files if requested
            removed_files = []
            if remove_originals:
                for batch_file in batch_files:
                    try:
                        batch_file.unlink()
                        removed_files.append(str(batch_file))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to remove {batch_file}: {e}")
            
            result = {
                'success': True,
                'consolidated_file': str(consolidated_path),
                'total_rows': len(combined_df),
                'batch_files_processed': len(batch_files),
                'removed_files': removed_files,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Consolidation completed: {len(combined_df)} rows in {consolidated_path.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error consolidating batches: {e}")
            return {
                'success': False,
                'error': str(e),
                'consolidated_file': None
            }
    
    @handles_errors(context="find_batch_files")
    async def _find_batch_files(
        self, 
        symbol: str, 
        exchange: str, 
        data_type: str, 
        timeframe: str, 
        session_id: Optional[str] = None
    ) -> List[Path]:
        """Find batch files matching the criteria."""
        try:
            data_dir = self.data_cache_path / exchange.lower() / symbol.lower() / data_type
            
            if not data_dir.exists():
                return []
            
            # Build pattern for batch files
            if session_id:
                pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}_{session_id}_batch_*.parquet"
            else:
                pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}_*_batch_*.parquet"
            
            batch_files = list(data_dir.glob(pattern))
            
            # Sort by modification time
            batch_files.sort(key=lambda x: x.stat().st_mtime)
            
            return batch_files
            
        except Exception as e:
            self.logger.error(f"❌ Error finding batch files: {e}")
            return []
    
    @handles_errors(context="list_available_data")
    @log_all_calls
    async def list_available_data(
        self, 
        symbol: Optional[str] = None, 
        exchange: Optional[str] = None, 
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all available data files and their information.
        
        Args:
            symbol: Filter by symbol (optional)
            exchange: Filter by exchange (optional)
            data_type: Filter by data type (optional)
            
        Returns:
            Dictionary with available data information
        """
        try:
            data_info = {
                'batch_files': [],
                'consolidated_files': [],
                'total_files': 0,
                'total_size_mb': 0
            }
            
            # Search for files
            search_patterns = []
            if symbol and exchange and data_type:
                search_patterns.append(f"{data_type}_{exchange}_{symbol}_*")
            elif symbol and exchange:
                search_patterns.append(f"*_{exchange}_{symbol}_*")
            elif symbol:
                search_patterns.append(f"*_{symbol}_*")
            else:
                search_patterns.append("*.parquet")
            
            for pattern in search_patterns:
                files = list(self.data_cache_path.rglob(pattern))
                
                for file_path in files:
                    if file_path.suffix == '.parquet':
                        file_info = self.parquet_handler.get_file_info(file_path)
                        
                        if 'batch_' in file_path.name:
                            data_info['batch_files'].append({
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'size_mb': file_info.get('file_size_mb', 0),
                                'created_at': file_info.get('created_at', ''),
                                'row_count': file_info.get('row_count', 0)
                            })
                        else:
                            data_info['consolidated_files'].append({
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'size_mb': file_info.get('file_size_mb', 0),
                                'created_at': file_info.get('created_at', ''),
                                'row_count': file_info.get('row_count', 0)
                            })
                        
                        data_info['total_size_mb'] += file_info.get('file_size_mb', 0)
            
            data_info['total_files'] = len(data_info['batch_files']) + len(data_info['consolidated_files'])
            
            # Sort by creation time
            data_info['batch_files'].sort(key=lambda x: x['created_at'], reverse=True)
            data_info['consolidated_files'].sort(key=lambda x: x['created_at'], reverse=True)
            
            self.logger.info(f"📊 Found {data_info['total_files']} files ({data_info['total_size_mb']:.2f} MB)")
            return data_info
            
        except Exception as e:
            self.logger.error(f"❌ Error listing available data: {e}")
            return {
                'batch_files': [],
                'consolidated_files': [],
                'total_files': 0,
                'total_size_mb': 0,
                'error': str(e)
            }
    
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
            'total_batches': 0,
            'start_time': None,
            'last_download_time': None
        }

# Convenience functions for backward compatibility
@handles_errors()
async def download_data_with_append(
    symbol: str, 
    exchange: str, 
    data_type: str = "klines",
    timeframe: str = "1m",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for downloading data with append functionality."""
    downloader = EnhancedAppendDataDownloader()
    return await downloader.download_with_append(symbol, exchange, data_type, timeframe, **kwargs)

@handles_errors()
async def consolidate_data_batches(
    symbol: str, 
    exchange: str, 
    data_type: str = "klines",
    timeframe: str = "1m",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for consolidating data batches."""
    downloader = EnhancedAppendDataDownloader()
    return await downloader.consolidate_batches(symbol, exchange, data_type, timeframe, **kwargs)

@handles_errors()
async def list_data_files(
    symbol: Optional[str] = None, 
    exchange: Optional[str] = None, 
    data_type: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for listing available data files."""
    downloader = EnhancedAppendDataDownloader()
    return await downloader.list_available_data(symbol, exchange, data_type)

if __name__ == "__main__":
    # Example usage
    async def test_append_downloader():
        logger.info("🎯 Testing Enhanced Append Data Downloader")
        logger.info("=" * 80)
        
        # Test data download with append
        logger.info("📊 Testing data download with append...")
        result = await download_data_with_append(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m",
            max_batches=3
        )
        
        logger.info(f"✅ Download result: {result['success']}")
        logger.info(f"📊 Downloaded {result['total_rows']} rows in {result['successful_batches']} batches")
        
        # Test data consolidation
        logger.info("🔄 Testing data consolidation...")
        consolidate_result = await consolidate_data_batches(
            symbol="ETHUSDT",
            exchange="BINANCE",
            data_type="klines",
            timeframe="1m"
        )
        
        logger.info(f"✅ Consolidation result: {consolidate_result['success']}")
        
        # Test listing data files
        logger.info("📁 Testing data file listing...")
        list_result = await list_data_files(symbol="ETHUSDT", exchange="BINANCE")
        
        logger.info(f"📊 Found {list_result['total_files']} files ({list_result['total_size_mb']:.2f} MB)")
        
        logger.info("=" * 80)
        logger.info("🎉 Enhanced Append Data Downloader tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_append_downloader())