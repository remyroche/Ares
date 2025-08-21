"""
Missing Data Downloader and Gap Filler

Automatically downloads missing data and fills gaps:
1. Downloads missing days of aggtrades (up to 2 days ago)
2. Downloads missing months of klines (up to 2 days ago) 
3. Downloads missing months of futures (up to 2 days ago)
4. Fills gaps over 10 seconds in aggtrades data
5. Integrates new data without deleting pre-existing data
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import aiofiles
import aiohttp
import asyncio
import os
import sys
import ssl
import certifi
import zipfile
import io
import traceback
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.exchange.binance import BinanceExchange
from src.utils.centralized_decorators import (
    with_tracing_span,
    handle_errors,
    validate_data_quality,
    validate_data_structure,
    guard_dataframe_nulls,
    optimize_memory_usage,
    comprehensive_data_validation,
    secure_data_processing
)

try:
    from .data_gap_detector import DataGapDetector
except ImportError:
    DataGapDetector = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild("MissingDataDownloader")

class MissingDataDownloaderAndGapFiller:
    """Downloads missing data and fills gaps automatically"""
    

    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        
        # Initialize exchange connection
        try:
            self.exchange = BinanceExchange(CONFIG)
        except ImportError:
            # Fallback configuration
            fallback_config = {
                "EXCHANGE": "BINANCE",
                "API_KEY": "",
                "API_SECRET": "",
                "TESTNET": True
            }
            self.exchange = BinanceExchange(fallback_config)
        
        # Download limits and retry settings
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.rate_limit_delay = 0.1  # seconds between requests
        
        # Gap detection settings
        self.min_gap_seconds = 10
        
        # Exchange initialization flag
        self._exchange_initialized = False
        
    async def _ensure_exchange_initialized(self) -> bool:
        """Ensure the exchange is properly initialized"""
        if not self._exchange_initialized:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                logger.info("🔧 Initializing Binance exchange connection...")
                success = await self.exchange.initialize()
                if success:
                    self._exchange_initialized = True
                    logger.info("✅ Binance exchange initialized successfully")
                    return True
                else:
                    logger.error("❌ Failed to initialize Binance exchange")
                    return False
            except Exception as e:
                logger.error(f"❌ Error initializing exchange: {e}")
                return False
        return True
        
    @secure_data_processing
    @with_tracing_span("get_current_timestamp")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError),
        default_return=datetime.now(),
        context="missing_data_downloader.get_current_timestamp"
    )

    def get_current_timestamp(self) -> datetime:
        """Get current timestamp from exchange to determine 'today'"""
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Get server time from exchange
            server_time = self.exchange.get_server_time()
            if server_time:
                return datetime.fromtimestamp(server_time / 1000)
            else:
                # Fallback to local time
                return datetime.now()
        except Exception as e:
            logger.warning(f"⚠️ Could not get server time, using local time: {e}")
            return datetime.now()
    
    @validate_data_structure
    @with_tracing_span("identify_missing_data")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return={'symbol': '', 'exchange': '', 'end_date': None, 'missing_aggtrades_days': [], 
                       'missing_klines_months': [], 'missing_futures_months': []},
        context="missing_data_downloader.identify_missing_data"
    )

    def identify_missing_data(self, symbol: str, exchange: str, 
                            end_date: Optional[datetime] = None) -> Dict:
        """
        Identify all missing data periods
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            end_date: End date for analysis (default: 2 days ago)
            
        Returns:
            Dictionary with missing data information
        """
        if end_date is None:
            current_time = self.get_current_timestamp()
            end_date = current_time - timedelta(days=2)
        
        start_date = end_date - timedelta(days=365*2)  # 2 years back
        
        logger.info(f"🔍 Identifying missing data for {exchange}_{symbol} from {start_date.date()} to {end_date.date()}")
        
        # Import gap detector to reuse existing logic
        gap_detector = DataGapDetector(str(self.data_cache_path))
        
        # Get missing data periods
        missing_data = gap_detector.detect_missing_data(symbol, exchange, start_date, end_date)
        
        # Get aggtrades gaps
        aggtrades_gaps = gap_detector.detect_aggtrades_gaps(symbol, exchange, self.min_gap_seconds)
        
        return {
            'symbol': symbol,
            'exchange': exchange,
            'start_date': start_date,
            'end_date': end_date,
            'missing_aggtrades_days': missing_data['missing_aggtrades_days'],
            'missing_klines_months': missing_data['missing_klines_months'],
            'missing_futures_months': missing_data['missing_futures_months'],
            'aggtrades_gaps': aggtrades_gaps,
            'total_missing_periods': (
                len(missing_data['missing_aggtrades_days']) +
                len(missing_data['missing_klines_months']) +
                len(missing_data['missing_futures_months'])
            )
        }
    
    @optimize_memory_usage
    @with_tracing_span("download_missing_aggtrades")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'downloaded_days': 0, 'failed_days': 0, 'total_rows': 0, 'errors': ['Download failed']},
        context="missing_data_downloader.download_missing_aggtrades"
    )
    async def download_missing_aggtrades(self, symbol: str, exchange: str,
                                       missing_days: List[datetime]) -> Dict:
        """
        Download missing aggtrades data for specific days
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            missing_days: List of missing dates
            
        Returns:
            Dictionary with download results
        """
        # Ensure exchange is initialized
        if not await self._ensure_exchange_initialized():
            return {
                'downloaded_days': 0,
                'failed_days': 0,
                'total_rows': 0,
                'errors': ['Failed to initialize exchange connection']
            }
            
        logger.info(f"📥 Downloading {len(missing_days)} missing aggtrades days for {exchange}_{symbol}")
        
        results = {
            'downloaded_days': 0,
            'failed_days': 0,
            'total_rows': 0,
            'errors': []
        }
        
        for date in missing_days:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                logger.info(f"📥 Downloading aggtrades for {date.date()}")
                
                # Download aggtrades for the day
                aggtrades_data = await self.exchange.get_aggregate_trades(
                    symbol=symbol,
                    start_time=date,
                    end_time=date + timedelta(days=1)
                )
                
                if aggtrades_data and len(aggtrades_data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(aggtrades_data)
                    
                    # Ensure proper column names and types
                    df = self._standardize_aggtrades_format(df)
                    
                    # Save to file
                    filename = f"aggtrades_{exchange}_{symbol}_{date.strftime('%Y-%m-%d')}.parquet"
                    file_path = self.data_cache_path / filename
                    
                    df.to_parquet(file_path, compression="zstd", index=False)
                    
                    results['downloaded_days'] += 1
                    results['total_rows'] += len(df)
                    
                    logger.info(f"✅ Downloaded {len(df)} aggtrades for {date.date()}")
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                else:
                    logger.warning(f"⚠️ No aggtrades data available for {date.date()}")
                    results['failed_days'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error downloading aggtrades for {date.date()}: {e}")
                results['failed_days'] += 1
                results['errors'].append(f"{date.date()}: {e}")
        
        logger.info(f"📊 Aggtrades download complete: {results['downloaded_days']} days, {results['total_rows']} rows")
        return results
    
    @optimize_memory_usage
    @with_tracing_span("download_missing_klines")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'downloaded_months': 0, 'failed_months': 0, 'total_rows': 0, 'errors': ['Download failed']},
        context="missing_data_downloader.download_missing_klines"
    )
    async def download_missing_klines(self, symbol: str, exchange: str,
                                    missing_months: List[datetime]) -> Dict:
        """
        Download missing klines data for specific months
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            missing_months: List of missing months
            
        Returns:
            Dictionary with download results
        """
        # Ensure exchange is initialized
        if not await self._ensure_exchange_initialized():
            return {
                'downloaded_months': 0,
                'failed_months': 0,
                'total_rows': 0,
                'errors': ['Failed to initialize exchange connection']
            }
            
        logger.info(f"📥 Downloading {len(missing_months)} missing klines months for {exchange}_{symbol}")
        
        results = {
            'downloaded_months': 0,
            'failed_months': 0,
            'total_rows': 0,
            'errors': []
        }
        
        for month_date in missing_months:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                logger.info(f"📥 Downloading klines for {month_date.strftime('%Y-%m')}")
                
                # Calculate month boundaries
                start_date = month_date.replace(day=1)
                if month_date.month == 12:
                    end_date = month_date.replace(year=month_date.year + 1, month=1)
                else:
                    end_date = month_date.replace(month=month_date.month + 1)
                
                # Download 1-minute klines for the month
                klines_data = await self.exchange.get_klines(
                    symbol=symbol,
                    interval='1m',
                    start_time=start_date,
                    end_time=end_date
                )
                
                if klines_data and len(klines_data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(klines_data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
                    ])
                    
                    # Keep only required columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Convert numeric columns
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove any rows with NaN values
                    df = df.dropna()
                    
                    if len(df) > 0:
                        # Save to file
                        filename = f"klines_{exchange}_{symbol}_1m_{month_date.strftime('%Y-%m')}.parquet"
                        file_path = self.data_cache_path / filename
                        
                        df.to_parquet(file_path, compression="zstd", index=False)
                        
                        results['downloaded_months'] += 1
                        results['total_rows'] += len(df)
                        
                        logger.info(f"✅ Downloaded {len(df)} klines for {month_date.strftime('%Y-%m')}")
                        
                        # Rate limiting
                        await asyncio.sleep(self.rate_limit_delay)
                    else:
                        logger.warning(f"⚠️ No valid klines data for {month_date.strftime('%Y-%m')}")
                        results['failed_months'] += 1
                else:
                    logger.warning(f"⚠️ No klines data available for {month_date.strftime('%Y-%m')}")
                    results['failed_months'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error downloading klines for {month_date.strftime('%Y-%m')}: {e}")
                results['failed_months'] += 1
                results['errors'].append(f"{month_date.strftime('%Y-%m')}: {e}")
        
        logger.info(f"📊 Klines download complete: {results['downloaded_months']} months, {results['total_rows']} rows")
        return results
    
    @optimize_memory_usage
    @with_tracing_span("download_missing_futures")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'downloaded_months': 0, 'failed_months': 0, 'total_rows': 0, 'errors': ['Download failed']},
        context="missing_data_downloader.download_missing_futures"
    )
    async def download_missing_futures(self, symbol: str, exchange: str,
                                     missing_months: List[datetime]) -> Dict:
        """
        Download missing futures data for specific months
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            missing_months: List of missing months
            
        Returns:
            Dictionary with download results
        """
        # Ensure exchange is initialized
        if not await self._ensure_exchange_initialized():
            return {
                'downloaded_months': 0,
                'failed_months': 0,
                'total_rows': 0,
                'errors': ['Failed to initialize exchange connection']
            }
            
        logger.info(f"📥 Downloading {len(missing_months)} missing futures months for {exchange}_{symbol}")
        
        results = {
            'downloaded_months': 0,
            'failed_months': 0,
            'total_rows': 0,
            'errors': []
        }
        
        for month_date in missing_months:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                logger.info(f"📥 Downloading futures for {month_date.strftime('%Y-%m')}")
                
                # Calculate month boundaries
                start_date = month_date.replace(day=1)
                if month_date.month == 12:
                    end_date = month_date.replace(year=month_date.year + 1, month=1)
                else:
                    end_date = month_date.replace(month=month_date.month + 1)
                
                # Download funding rate data for the month
                futures_data = await self.exchange.get_funding_rate(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if futures_data and len(futures_data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(futures_data)
                    
                    # Ensure proper column names and types
                    df = self._standardize_futures_format(df)
                    
                    if len(df) > 0:
                        # Save to file
                        filename = f"futures_{exchange}_{symbol}_{month_date.strftime('%Y-%m')}.parquet"
                        file_path = self.data_cache_path / filename
                        
                        df.to_parquet(file_path, compression="zstd", index=False)
                        
                        results['downloaded_months'] += 1
                        results['total_rows'] += len(df)
                        
                        logger.info(f"✅ Downloaded {len(df)} futures records for {month_date.strftime('%Y-%m')}")
                        
                        # Rate limiting
                        await asyncio.sleep(self.rate_limit_delay)
                    else:
                        logger.warning(f"⚠️ No valid futures data for {month_date.strftime('%Y-%m')}")
                        results['failed_months'] += 1
                else:
                    logger.warning(f"⚠️ No futures data available for {month_date.strftime('%Y-%m')}")
                    results['failed_months'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error downloading futures for {month_date.strftime('%Y-%m')}: {e}")
                results['failed_months'] += 1
                results['errors'].append(f"{month_date.strftime('%Y-%m')}: {e}")
        
        logger.info(f"📊 Futures download complete: {results['downloaded_months']} months, {results['total_rows']} rows")
        return results
    
    @comprehensive_data_validation
    @with_tracing_span("fill_single_gap")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'success': False, 'error': 'Gap filling failed', 'rows_added': 0},
        context="missing_data_downloader.fill_single_gap"
    )
    async def fill_single_gap(self, symbol: str, exchange: str, 
                             gap_info: Dict) -> Dict:
        """
        Fill a single gap in aggtrades data by downloading missing period
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            gap_info: Gap information dictionary
            
        Returns:
            Dictionary with gap filling result
        """
        # Ensure exchange is initialized
        if not await self._ensure_exchange_initialized():
            return {
                'success': False,
                'error': 'Failed to initialize exchange connection',
                'rows_added': 0
            }
            
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            gap_start = gap_info['gap_start']
            gap_end = gap_info['gap_end']
            file_name = gap_info['file']
            
            # Convert datetime to milliseconds
            if isinstance(gap_start, str):
                gap_start = pd.to_datetime(gap_start)
            if isinstance(gap_end, str):
                gap_end = pd.to_datetime(gap_end)
            
            start_time_ms = int(gap_start.timestamp() * 1000)
            end_time_ms = int(gap_end.timestamp() * 1000)
            
            missing_data = None
            
            # Method 1: Try regular Binance API first
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                logger.info(f"🔍 Trying regular Binance API for gap: {gap_start} to {gap_end}")
                missing_data = await self.exchange.get_aggregate_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms
                )
                
                if missing_data and len(missing_data) > 0:
                    logger.info(f"✅ Regular API successful: {len(missing_data)} trades found")
                else:
                    logger.info("⚠️ Regular API returned no data, trying Binance Vision...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Regular API failed: {e}, trying Binance Vision...")
            
            # Method 2: Try Binance Vision if regular API failed or returned no data
            if not missing_data or len(missing_data) == 0:
                try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                    logger.info(f"🔍 Trying Binance Vision for gap: {gap_start} to {gap_end}")
                    missing_data = await self._fetch_aggtrades_from_binance_vision(
                        symbol=symbol,
                        gap_start=gap_start,
                        gap_end=gap_end,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms
                    )
                    
                    if missing_data and len(missing_data) > 0:
                        logger.info(f"✅ Binance Vision successful: {len(missing_data)} trades found")
                    else:
                        logger.warning("⚠️ Binance Vision also returned no data")
                        
                except Exception as e:
                    logger.error(f"❌ Binance Vision failed: {e}")
            
            if missing_data and len(missing_data) > 0:
                # Convert to DataFrame
                df_missing = pd.DataFrame(missing_data)
                df_missing = self._standardize_aggtrades_format(df_missing)
                
                # Load existing file
                file_path = self.data_cache_path / file_name
                if file_path.exists():
                    df_existing = pd.read_parquet(file_path)
                    
                    # Combine existing and new data
                    df_combined = pd.concat([df_existing, df_missing], ignore_index=True)
                    
                    # Sort by timestamp and remove duplicates
                    df_combined = df_combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                    
                    # Save back to file
                    df_combined.to_parquet(file_path, compression="zstd", index=False)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    return {
                        'success': True,
                        'rows_added': len(df_missing),
                        'gap_duration': gap_info['gap_duration_seconds']
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Could not find existing file: {file_name}',
                        'rows_added': 0
                    }
            else:
                return {
                    'success': False,
                    'error': 'No data available to fill gap (tried both regular API and Binance Vision)',
                    'rows_added': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'rows_added': 0
            }

    @comprehensive_data_validation
    @with_tracing_span("fill_aggtrades_gaps")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'filled_gaps': 0, 'failed_gaps': 0, 'total_rows': 0, 'errors': ['Gap filling failed']},
        context="missing_data_downloader.fill_aggtrades_gaps"
    )
    async def fill_aggtrades_gaps(self, symbol: str, exchange: str, 
                                gaps: List[Dict]) -> Dict:
        """
        Fill gaps in aggtrades data by downloading missing periods
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            gaps: List of gap information dictionaries
            
        Returns:
            Dictionary with gap filling results
        """
        # Ensure exchange is initialized
        if not await self._ensure_exchange_initialized():
            return {
                'filled_gaps': 0,
                'failed_gaps': 0,
                'total_rows_added': 0,
                'errors': ['Failed to initialize exchange connection']
            }
            
        logger.info(f"🔧 Filling {len(gaps)} aggtrades gaps for {exchange}_{symbol}")
        
        results = {
            'filled_gaps': 0,
            'failed_gaps': 0,
            'total_rows_added': 0,
            'errors': []
        }
        
        for gap in gaps:
            try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
                gap_start = gap['gap_start']
                gap_end = gap['gap_end']
                file_name = gap['file']
                
                logger.info(f"🔧 Filling gap in {file_name}: {gap_start} to {gap_end}")
                
                # Download missing data for the gap period
                # Convert datetime to milliseconds
                if isinstance(gap_start, str):
                    gap_start = pd.to_datetime(gap_start)
                if isinstance(gap_end, str):
                    gap_end = pd.to_datetime(gap_end)
                
                start_time_ms = int(gap_start.timestamp() * 1000)
                end_time_ms = int(gap_end.timestamp() * 1000)
                
                missing_data = await self.exchange.get_aggregate_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms
                )
                
                if missing_data and len(missing_data) > 0:
                    # Convert to DataFrame
                    df_missing = pd.DataFrame(missing_data)
                    df_missing = self._standardize_aggtrades_format(df_missing)
                    
                    # Load existing file
                    file_path = self.data_cache_path / file_name
                    if file_path.exists():
                        df_existing = pd.read_parquet(file_path)
                        
                        # Combine existing and new data
                        df_combined = pd.concat([df_existing, df_missing], ignore_index=True)
                        
                        # Sort by timestamp and remove duplicates
                        df_combined = df_combined.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                        
                        # Save back to file
                        df_combined.to_parquet(file_path, compression="zstd", index=False)
                        
                        results['filled_gaps'] += 1
                        results['total_rows_added'] += len(df_missing)
                        
                        logger.info(f"✅ Filled gap in {file_name}: added {len(df_missing)} rows")
                    else:
                        logger.warning(f"⚠️ Could not find existing file: {file_name}")
                        results['failed_gaps'] += 1
                        
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                else:
                    logger.warning(f"⚠️ No data available to fill gap in {file_name}")
                    results['failed_gaps'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error filling gap in {gap.get('file', 'unknown')}: {e}")
                results['failed_gaps'] += 1
                results['errors'].append(f"{gap.get('file', 'unknown')}: {e}")
        
        logger.info(f"📊 Gap filling complete: {results['filled_gaps']} gaps filled, {results['total_rows_added']} rows added")
        return results
    
    @validate_data_quality
    @with_tracing_span("_standardize_aggtrades_format")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError),
        default_return=pd.DataFrame(),
        context="missing_data_downloader._standardize_aggtrades_format"
    )

    def _standardize_aggtrades_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize aggtrades DataFrame format"""
        # Expected columns
        expected_columns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker']
        
        # Rename columns if needed
        column_mapping = {
            'a': 'agg_trade_id',
            'p': 'price',
            'q': 'quantity',
            'f': 'first_trade_id',
            'l': 'last_trade_id',
            'T': 'timestamp',
            'm': 'is_buyer_maker'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        for col in expected_columns:
            if col not in df.columns:
                if col == 'is_buyer_maker':
                    df[col] = False  # Default value
                else:
                    df[col] = 0  # Default value
        
        # Convert data types
        df['agg_trade_id'] = pd.to_numeric(df['agg_trade_id'], errors='coerce').fillna(0).astype('int64')
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype('float64')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0).astype('float64')
        df['first_trade_id'] = pd.to_numeric(df['first_trade_id'], errors='coerce').fillna(0).astype('int64')
        df['last_trade_id'] = pd.to_numeric(df['last_trade_id'], errors='coerce').fillna(0).astype('int64')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['is_buyer_maker'] = df['is_buyer_maker'].astype('bool')
        
        # Remove any rows with NaN values in critical columns
        critical_columns = ['timestamp', 'price', 'quantity']
        df = df.dropna(subset=critical_columns)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df[expected_columns]
    
    @validate_data_quality
    @with_tracing_span("_standardize_futures_format")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError),
        default_return=pd.DataFrame(),
        context="missing_data_downloader._standardize_futures_format"
    )

    def _standardize_futures_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize futures DataFrame format"""
        # Expected columns
        expected_columns = ['timestamp', 'fundingRate']
        
        # Rename columns if needed
        column_mapping = {
            'fundingTime': 'timestamp',
            'fundingRate': 'fundingRate'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        for col in expected_columns:
            if col not in df.columns:
                if col == 'fundingRate':
                    df[col] = 0.0  # Default value
                else:
                    df[col] = 0  # Default value
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce').fillna(0.0).astype('float64')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df[expected_columns]
    
    @comprehensive_data_validation
    @with_tracing_span("download_all_missing_data")
    @handle_errors
    async def download_all_missing_data(self, symbol: str, exchange: str,
                                      end_date: Optional[datetime] = None) -> Dict:
        """
        Download all missing data and fill gaps
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            end_date: End date for analysis (default: 2 days ago)
            
        Returns:
            Dictionary with comprehensive download results
        """
        logger.info(f"🚀 STARTING COMPREHENSIVE MISSING DATA DOWNLOAD FOR {exchange}_{symbol}")
        logger.info("=" * 80)
        
        # Get current timestamp for "today" reference
        current_time = self.get_current_timestamp()
        if end_date is None:
            end_date = current_time - timedelta(days=2)
        
        logger.info(f"📅 Using current time: {current_time}")
        logger.info(f"📅 Analysis end date: {end_date}")
        
        # Identify missing data
        missing_data = self.identify_missing_data(symbol, exchange, end_date)
        
        results = {
            'symbol': symbol,
            'exchange': exchange,
            'current_time': current_time,
            'analysis_end_date': end_date,
            'missing_data': missing_data,
            'download_results': {},
            'gap_filling_results': {},
            'success': True,
            'errors': []
        }
        
        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Download missing aggtrades
            if missing_data['missing_aggtrades_days']:
                logger.info("📥 STEP 1: DOWNLOADING MISSING AGGTRADES")
                logger.info("-" * 60)
                
                aggtrades_results = await self.download_missing_aggtrades(
                    symbol, exchange, missing_data['missing_aggtrades_days']
                )
                results['download_results']['aggtrades'] = aggtrades_results
                
                if aggtrades_results['failed_days'] > 0:
                    results['errors'].extend(aggtrades_results['errors'])
            
            # Download missing klines
            if missing_data['missing_klines_months']:
                logger.info("📥 STEP 2: DOWNLOADING MISSING KLINES")
                logger.info("-" * 60)
                
                klines_results = await self.download_missing_klines(
                    symbol, exchange, missing_data['missing_klines_months']
                )
                results['download_results']['klines'] = klines_results
                
                if klines_results['failed_months'] > 0:
                    results['errors'].extend(klines_results['errors'])
            
            # Download missing futures
            if missing_data['missing_futures_months']:
                logger.info("📥 STEP 3: DOWNLOADING MISSING FUTURES")
                logger.info("-" * 60)
                
                futures_results = await self.download_missing_futures(
                    symbol, exchange, missing_data['missing_futures_months']
                )
                results['download_results']['futures'] = futures_results
                
                if futures_results['failed_months'] > 0:
                    results['errors'].extend(futures_results['errors'])
            
            # Fill aggtrades gaps
            if missing_data['aggtrades_gaps']:
                logger.info("🔧 STEP 4: FILLING AGGTRADES GAPS")
                logger.info("-" * 60)
                
                gap_results = await self.fill_aggtrades_gaps(
                    symbol, exchange, missing_data['aggtrades_gaps']
                )
                results['gap_filling_results'] = gap_results
                
                if gap_results['failed_gaps'] > 0:
                    results['errors'].extend(gap_results['errors'])
            
            # Generate summary report
            logger.info("📊 STEP 5: GENERATING SUMMARY REPORT")
            logger.info("-" * 60)
            
            report = self.generate_download_report(results)
            results['report'] = report
            
            logger.info("=" * 80)
            if results['success']:
                logger.info("🎉 MISSING DATA DOWNLOAD COMPLETED SUCCESSFULLY!")
            else:
                logger.error("❌ MISSING DATA DOWNLOAD COMPLETED WITH ERRORS!")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive download: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            return results
    
    @with_tracing_span("generate_download_report")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, AttributeError),
        default_return="❌ ERROR: Failed to generate download report",
        context="missing_data_downloader.generate_download_report"
    )

    def generate_download_report(self, results: Dict) -> str:
        """Generate a comprehensive download report"""
        report = f"""
📥 MISSING DATA DOWNLOAD REPORT FOR {results['exchange']}_{results['symbol']}
{'='*80}

📅 TIMING INFORMATION:
• Current Time: {results['current_time']}
• Analysis End Date: {results['analysis_end_date']}

📊 MISSING DATA SUMMARY:
• Missing Aggtrades Days: {len(results['missing_data']['missing_aggtrades_days'])}
• Missing Klines Months: {len(results['missing_data']['missing_klines_months'])}
• Missing Futures Months: {len(results['missing_data']['missing_futures_months'])}
• Aggtrades Gaps > 10s: {len(results['missing_data']['aggtrades_gaps'])}

📥 DOWNLOAD RESULTS:
"""
        
        # Aggtrades download results
        if 'aggtrades' in results['download_results']:
            aggtrades = results['download_results']['aggtrades']
            report += f"""
• Aggtrades:
  - Downloaded Days: {aggtrades['downloaded_days']}
  - Failed Days: {aggtrades['failed_days']}
  - Total Rows: {aggtrades['total_rows']}
"""
        
        # Klines download results
        if 'klines' in results['download_results']:
            klines = results['download_results']['klines']
            report += f"""
• Klines:
  - Downloaded Months: {klines['downloaded_months']}
  - Failed Months: {klines['failed_months']}
  - Total Rows: {klines['total_rows']}
"""
        
        # Futures download results
        if 'futures' in results['download_results']:
            futures = results['download_results']['futures']
            report += f"""
• Futures:
  - Downloaded Months: {futures['downloaded_months']}
  - Failed Months: {futures['failed_months']}
  - Total Rows: {futures['total_rows']}
"""
        
        # Gap filling results
        if results['gap_filling_results']:
            gaps = results['gap_filling_results']
            report += f"""
🔧 GAP FILLING RESULTS:
• Filled Gaps: {gaps['filled_gaps']}
• Failed Gaps: {gaps['failed_gaps']}
• Total Rows Added: {gaps['total_rows_added']}
"""
        
        # Errors
        if results['errors']:
            report += f"""
❌ ERRORS:
{chr(10).join(f'• {error}' for error in results['errors'])}
"""
        
        report += f"""
{'='*80}
"""
        
        return report

    async def _fetch_aggtrades_from_binance_vision(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        start_time_ms: int,
        end_time_ms: int,
        market_segment: str = "um",
    ) -> List[Dict]:
        """Download aggregated trades from Binance Data (binance.vision) for a specific gap period.

        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            gap_start: Start datetime of the gap
            gap_end: End datetime of the gap
            start_time_ms: Lower bound (inclusive) in ms to filter rows
            end_time_ms: Upper bound (exclusive) in ms to filter rows
            market_segment: 'um' for USDT-M futures, 'cm' for COIN-M futures

        Returns:
            List of trade dicts with keys matching Binance aggTrades API ('a','p','q','f','l','T','m').
        """
        base_url = "https://data.binance.vision"
        date_str = gap_start.strftime("%Y-%m-%d")
        # Futures USDT-M (fapi) dataset path
        path = f"data/futures/{market_segment}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        url = f"{base_url}/{path}"

        try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
            # Create aiohttp session if not exists
            if not hasattr(self, 'session') or self.session is None:
                self.session = aiohttp.ClientSession()

            # Use certifi CA bundle to avoid SSL verification issues
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            async with self.session.get(url, ssl=ssl_context) as resp:
                if resp.status != 200:
                    logger.info(
                        f"Binance Vision: no file for {symbol} {date_str} (status {resp.status})",
                    )
                    return []
                content = await resp.read()

            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Pick first CSV entry
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.warning(
                        f"Binance Vision: archive for {symbol} {date_str} has no CSV entries",
                    )
                    return []
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=["a", "p", "q", "f", "l", "T", "m", "M"],
                        low_memory=False,
                    )

            if df.empty:
                return []

            # Coerce types to expected numeric/bool
            for col in ["a", "f", "l", "T"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["p", "q"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # Normalize boolean 'm'
            df["m"] = (
                df["m"]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    },
                )
                .fillna(False)
                .astype("boolean")
            )

            # Drop rows with invalid timestamps
            df = df.dropna(subset=["T"])

            # Filter to the effective time window
            df = df[(df["T"] >= start_time_ms) & (df["T"] < end_time_ms)]
            if df.empty:
                return []

            # Convert to list of dicts compatible with _process_aggtrades_data
            return df[["a", "p", "q", "f", "l", "T", "m"]].to_dict(
                orient="records",
            )
        except Exception as e:
            error_details = traceback.format_exc()
            logger.warning(
                f"Binance Vision fallback failed for {symbol} {date_str}: {e}\n{error_details}",
            )
            return []

    @with_tracing_span("regenerate_timeframe_files")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'success': False, 'errors': ['Failed to regenerate timeframe files']},
        context="missing_data_downloader.regenerate_timeframe_files"
    )
    async def regenerate_timeframe_files(self, symbol: str, exchange: str, 
                                       timeframes: Optional[List[str]] = None) -> Dict:
        """
        Regenerate all timeframe files after data has been updated/fixed
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to regenerate (default: all supported)
            
        Returns:
            Dictionary with regeneration results
        """
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h", "4h"]
            
        logger.info(f"🔄 Regenerating timeframe files for {exchange}_{symbol}: {timeframes}")
        
        results = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframes': timeframes,
            'regenerated_files': {},
            'failed_timeframes': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Import the data resampler
            from .data_resampler import DataPreparation
            
            # Initialize data preparation
            data_prep = DataPreparation(str(self.data_cache_path))
            
            # Get the latest date range from 1m data
            klines_files = list(self.data_cache_path.glob(f"klines_{exchange}_{symbol}_1m_*.parquet"))
            if not klines_files:
                logger.error(f"❌ No 1m klines files found for {exchange}_{symbol}")
                results['success'] = False
                results['errors'].append("No 1m klines files found")
                return results
            
            # Load and combine all 1m data to get the full date range
            all_1m_data = []
            for file_path in klines_files:
                try:
                    df = pd.read_parquet(file_path)
                    all_1m_data.append(df)
                except Exception as e:
                    logger.warning(f"⚠️ Error reading {file_path}: {e}")
                    continue
            
            if not all_1m_data:
                logger.error(f"❌ No valid 1m data found for {exchange}_{symbol}")
                results['success'] = False
                results['errors'].append("No valid 1m data found")
                return results
            
            # Combine all 1m data
            combined_1m = pd.concat(all_1m_data, ignore_index=True)
            combined_1m = combined_1m.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            logger.info(f"📊 Combined 1m data: {len(combined_1m)} rows from {combined_1m['timestamp'].min()} to {combined_1m['timestamp'].max()}")
            
            # Regenerate each timeframe
            for timeframe in timeframes:
                try:
                    logger.info(f"🔄 Regenerating {timeframe} timeframe...")
                    
                    # Resample to the target timeframe
                    resampled_df = data_prep.resample_to_timeframe(combined_1m, timeframe)
                    
                    if len(resampled_df) == 0:
                        logger.warning(f"⚠️ No data after resampling to {timeframe}")
                        results['failed_timeframes'].append(timeframe)
                        continue
                    
                    # Save the resampled data
                    output_path = data_prep.save_resampled_data(resampled_df, symbol, exchange, timeframe)
                    
                    if output_path:
                        results['regenerated_files'][timeframe] = str(output_path)
                        logger.info(f"✅ Regenerated {timeframe}: {len(resampled_df)} rows -> {output_path}")
                    else:
                        logger.error(f"❌ Failed to save {timeframe} data")
                        results['failed_timeframes'].append(timeframe)
                        results['errors'].append(f"Failed to save {timeframe} data")
                        
                except Exception as e:
                    logger.error(f"❌ Error regenerating {timeframe}: {e}")
                    results['failed_timeframes'].append(timeframe)
                    results['errors'].append(f"{timeframe}: {e}")
            
            # Summary
            successful = len(results['regenerated_files'])
            failed = len(results['failed_timeframes'])
            
            logger.info(f"📊 Timeframe regeneration complete: {successful} successful, {failed} failed")
            
            if failed > 0:
                results['success'] = False
                
        except Exception as e:
            logger.error(f"❌ Error in timeframe regeneration: {e}")
            results['success'] = False
            results['errors'].append(f"General error: {e}")
            
        return results

    @with_tracing_span("fill_aggtrades_gaps_with_regeneration")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError, 
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={'success': False, 'errors': ['Failed to fill gaps and regenerate timeframes']},
        context="missing_data_downloader.fill_aggtrades_gaps_with_regeneration"
    )
    async def fill_aggtrades_gaps_with_regeneration(self, symbol: str, exchange: str, 
                                                  gaps: List[Dict], 
                                                  regenerate_timeframes: bool = True,
                                                  timeframes: Optional[List[str]] = None) -> Dict:
        """
        Fill gaps in aggtrades data and optionally regenerate all timeframe files
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            gaps: List of gap information dictionaries
            regenerate_timeframes: Whether to regenerate timeframe files after gap filling
            timeframes: List of timeframes to regenerate (default: all supported)
            
        Returns:
            Dictionary with gap filling and regeneration results
        """
        logger.info(f"🔧 Filling gaps and regenerating timeframes for {exchange}_{symbol}")
        
        # First, fill the gaps
        gap_results = await self.fill_aggtrades_gaps(symbol, exchange, gaps)
        
        results = {
            'symbol': symbol,
            'exchange': exchange,
            'gap_filling': gap_results,
            'timeframe_regeneration': None,
            'success': gap_results.get('filled_gaps', 0) > 0
        }
        
        # If gaps were filled and regeneration is requested, regenerate timeframes
        if regenerate_timeframes and gap_results.get('filled_gaps', 0) > 0:
            logger.info(f"🔄 Gaps were filled, regenerating timeframe files...")
            
            # Wait a moment to ensure files are written
            await asyncio.sleep(1)
            
            # Regenerate timeframe files
            timeframe_results = await self.regenerate_timeframe_files(symbol, exchange, timeframes)
            results['timeframe_regeneration'] = timeframe_results
            
            # Update overall success
            results['success'] = results['success'] and timeframe_results.get('success', False)
            
            logger.info(f"✅ Gap filling and timeframe regeneration complete")
        else:
            logger.info(f"ℹ️ Skipping timeframe regeneration (no gaps filled or regeneration disabled)")
            
        return results
