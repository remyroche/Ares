#!/usr / bin / env python3
""""""Missing Data Downloader and Gap Filler for Step1."""
"""
Downloads missing data and fills gaps automatically."""
""""""""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import required modules
try:
    except Exception as e:
        pass
    from src.utils.centralized_decorators import ()
        handle_errors,
        validate_data_quality,
        validate_data_structure,
        guard_dataframe_nulls,
        optimize_memory_usage,
        comprehensive_data_validation,
        secure_data_processing,
        with_tracing_span,
    
    from src.utils.logger import system_logger
    from src.training.steps.step1.data_gap_detector import DataGapDetector
    from src.exchange.binance_exchange import BinanceExchange
"
    # Configuration for Binance"""
    CONFIG = {}"""
        "API_KEY": "","""
        "API_SECRET": "","""
        "TESTNET"": True,"
    

except ImportError as e:
    # Fallback logger"
    import logging"""
    logging.basicConfig(level = logging.INFO)""""
    system_logger, logging.getLogger("MissingDataDownloaderFallback")""""
    system_logger.warning(f"⚠️ Some imports failed: {e}")

    # Fallback decorators
    def handle_errors(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def with_tracing_span(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def validate_data_quality(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def validate_data_structure(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def guard_dataframe_nulls(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def optimize_memory_usage(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def comprehensive_data_validation(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    def secure_data_processing(*args, **kwargs):
        def decorator(func):
            pass
        return func
        return decorator

    DataGapDetector, None
    BinanceExchange, None

    # Fallback logger"
    import logging"""
    logging.basicConfig(level = logging.INFO)""""
    system_logger, logging.getLogger("MissingDataDownloaderFallback")""
"""""
logger, system_logger.getChild("MissingDataDownloader")""
""
class MissingDataDownloaderAndGapFiller:"""
    """Downloads missing data and fills gaps automatically."""""
""""
    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path, Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok = True)

        # Initialize exchange connection
        if BinanceExchange:
            pass
        try:
            except Exception as e:
                pass"
        self.exchange, BinanceExchange(CONFIG)"""
        except Exception as e:""""
                logger.warning(f"Failed to initialize exchange: {e}")
        self.exchange, None
        else:
        self.exchange, None

        # Download limits and retry settings
        self.max_retries, 3
        self.retry_delay, 1.0  # seconds
        self.rate_limit_delay, 0.1  # seconds between requests

        # Gap detection settings
        self.min_gap_seconds, 10

        # Exchange initialization flag
        self._exchange_initialized, False

    @handle_errors()"
        exceptions=(OSError, ValueError, TypeError, KeyError),"""
        default_return = False,""""
        context="missing_data_downloader.ensure_exchange_initialized""
    """
    async def _ensure_exchange_initialized(self) -> bool:"""
        """Ensure the exchange is properly initialized."""""
        if not self._exchange_initialized:
            pass
        try:"
            except Exception as e:"""
                pass""""
                logger.info("🔧 Initializing Binance exchange connection...")
        if self.exchange:
                    success, await self.exchange.initialize()
        if success:"
            pass"""
        self._exchange_initialized, True""""
                        logger.info("✅ Binance exchange initialized successfully")"
        return True"""
                    else:""""
                        logger.error("❌ Failed to initialize Binance exchange")"
        return False"""
                else:""""
                    logger.warning("⚠️ No exchange available")"
        return False"""
        except Exception as e:""""
                logger.exception(f"❌ Error initializing exchange: {e}")
        return False"
        return True""
"""""
    @with_tracing_span("download_aggtrades_data")"
    @handle_errors()"""
        exceptions=(OSError, ValueError, TypeError, KeyError),""""
        default_return={"success": False, "error": "Download failed"},""""
        context="missing_data_downloader.download_aggtrades_data"
    
    async def download_aggtrades_data()"
        self, symbol: str, exchange: str, start_date: datetime, end_date: datetime"""
    ) -> dict:"""
        """"""Download aggtrades data for a specific date range.""

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date
            end_date: End date

        Returns:"
            Dictionary with download results""
"""
        """"""""""""""
        logger.info(f"📥 Downloading aggtrades data for {exchange}_{symbol}")
"
        if not await self._ensure_exchange_initialized():"""
            pass""""
        return {"success": False, "error": "Exchange not initialized"}"
"""
        results = {}"""
            "success": True,"""
            "downloaded_days": 0,"""
            "failed_days": 0,"""
            "total_rows": 0,"""
            "errors"": [],"
        

        # Generate list of dates to download
        current_date, start_date.date()
        dates_to_download = []
"
        while current_date <= end_date.date():"""
        # Check if file already exists""""
            filename, f"aggtrades_{exchange}_{symbol}_{current_date.strftime("%Y%m%d')}.parquet'
            file_path, self.data_cache_path / filename

        if not file_path.exists():'
                dates_to_download.append(current_date)'''
            else:''''
                logger.debug(f"📁 File already exists: {filename}")
"
            current_date += timedelta(days = 1)""
"""""
        logger.info(f"📊 Found {len(dates_to_download)} days to download")

        # Download data for each date
        for date in dates_to_download:
            pass
        try:
            except Exception as e:
                pass"
                success, await self._download_single_aggtrades_day(symbol, exchange, date)"""
        if success:""""
                    results["downloaded_days"] += 1"""
                else:""""
                    results["failed_days"] += 1""""
                    results["errors"].append(f"Failed to download {date}")

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)"
"""
        except Exception as e:""""
                results["failed_days"] += 1""""
                results["errors"].append(f"Error downloading {date}: {e}")""""
                logger.exception(f"❌ Error downloading {date}: {e}")"
"""
        # Count total rows""""
        results["total_rows"] = await self._count_aggtrades_rows(symbol, exchange)"
"""
        logger.info()""""
            f"📊 Download complete: {results["downloaded_days']} downloaded, ''''''''
            f"{results["failed_days']} failed, {results['total_rows']} total rows'
        

        return results

    async def _download_single_aggtrades_day()'
        self, symbol: str, exchange: str, date: datetime.date'''
    ) -> bool:''''
        """Download aggtrades data for a single day."""
        try:
            except Exception as e:"
                pass"""
        # Create filename""""
            filename, f"aggtrades_{exchange}_{symbol}_{date.strftime("%Y%m%d')}.parquet'
            file_path, self.data_cache_path / filename

        # Convert date to datetime for API calls
            start_time, datetime.combine(date, datetime.min.time())
            end_time, datetime.combine(date, datetime.max.time())

        # Download data using exchange API
        if self.exchange:
                data, await self.exchange.fetch_aggtrades()
                    symbol = symbol,
                    since = int(start_time.timestamp() * 1000),
                    limit = 1000,
                

        if data:
        # Convert to DataFrame
                    df, pd.DataFrame(data)
'
        # Standardize column names'''
                    column_mapping = {}''''
                        "a": "agg_trade_id","""
                        "p": "price","""
                        "q": "quantity","""
                        "f": "first_trade_id","""
                        "l": "last_trade_id","""
                        "T": "timestamp","""
                        "m": "is_buyer_maker",
                    

        if list(df.columns) != list(column_mapping.values()):"
                        df, df.rename(columns = column_mapping)""
""
        # Convert timestamp""""
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")"
"""
        # Save to parquet""""
                    df.to_parquet(file_path, compression="zstd", index = False)""
"""""
                    logger.info(f"✅ Downloaded {filename}: {len(df)} rows")"
        return True"""
                else:""""
                    logger.warning(f"⚠️ No data available for {date}")"
        return False"""
            else:""""
                logger.warning("⚠️ No exchange available for download")
        return False"
"""
        except Exception as e:""""
            logger.exception(f"❌ Error downloading {date}: {e}")
        return False"
"""
    async def _count_aggtrades_rows(self, symbol: str, exchange: str) -> int:"""
        """Count total rows in aggtrades files."""""
        try:"
            except Exception as e:"""
                pass""""
            pattern, f"aggtrades_{exchange}_{symbol}_*.parquet"
            files, list(self.data_cache_path.glob(pattern))

            total_rows, 0
        for file_path in files:
                df, pd.read_parquet(file_path)
                total_rows += len(df)
"
        return total_rows"""
        except Exception as e:""""
            logger.exception(f"❌ Error counting rows: {e}")"
        return 0""
"""""
    @with_tracing_span("download_klines_data")"
    @handle_errors()"""
        exceptions=(OSError, ValueError, TypeError, KeyError),""""
        default_return={"success": False, "error": "Download failed"},""""
        context="missing_data_downloader.download_klines_data"
    
    async def download_klines_data()"
        self, symbol: str, exchange: str, start_date: datetime, end_date: datetime"""
    ) -> dict:"""
        """"""Download klines data for a specific date range.""

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date
            end_date: End date

        Returns:"
            Dictionary with download results""
"""
        """"""""""""""
        logger.info(f"📥 Downloading klines data for {exchange}_{symbol}")
"
        if not await self._ensure_exchange_initialized():"""
            pass""""
        return {"success": False, "error": "Exchange not initialized"}"
"""
        results = {}"""
            "success": True,"""
            "downloaded_months": 0,"""
            "failed_months": 0,"""
            "total_rows": 0,"""
            "errors"": [],"
        

        # Generate list of months to download
        current_date, start_date.replace(day = 1)
        months_to_download = []
"
        while current_date <= end_date:"""
        # Check if file already exists""""
            filename, f"klines_{exchange}_{symbol}_1m_{current_date.strftime("%Y%m')}.parquet'
            file_path, self.data_cache_path / filename

        if not file_path.exists():'
                months_to_download.append(current_date)'''
            else:''''
                logger.debug(f"📁 File already exists: {filename}")

        # Move to next month
        if current_date.month == 12:
                current_date, current_date.replace(year = current_date.year + 1, month = 1)
            else:"
                current_date, current_date.replace(month = current_date.month + 1)""
"""""
        logger.info(f"📊 Found {len(months_to_download)} months to download")

        # Download data for each month
        for month in months_to_download:
            pass
        try:
            except Exception as e:
                pass"
                success, await self._download_single_klines_month(symbol, exchange, month)"""
        if success:""""
                    results["downloaded_months"] += 1"""
                else:""""
                    results["failed_months"] += 1""""
                    results["errors"].append(f"Failed to download {month}")

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)"
"""
        except Exception as e:""""
                results["failed_months"] += 1""""
                results["errors"].append(f"Error downloading {month}: {e}")""""
                logger.exception(f"❌ Error downloading {month}: {e}")"
"""
        # Count total rows""""
        results["total_rows"] = await self._count_klines_rows(symbol, exchange)"
"""
        logger.info()""""
            f"📊 Download complete: {results["downloaded_months']} downloaded, ''''''''
            f"{results["failed_months']} failed, {results['total_rows']} total rows'
        

        return results

    async def _download_single_klines_month()'
        self, symbol: str, exchange: str, month: datetime'''
    ) -> bool:''''
        """Download klines data for a single month."""
        try:
            except Exception as e:"
                pass"""
        # Create filename""""
            filename, f"klines_{exchange}_{symbol}_1m_{month.strftime("%Y%m')}.parquet'
            file_path, self.data_cache_path / filename

        # Calculate month boundaries
            start_time, month.replace(day = 1)
        if month.month == 12:
                end_time, month.replace(year = month.year + 1, month = 1, day = 1) - timedelta(seconds = 1)
            else:
                end_time, month.replace(month = month.month + 1, day = 1) - timedelta(seconds = 1)

        # Download data using exchange API
        if self.exchange:'
                data, await self.exchange.fetch_ohlcv()'''
                    symbol = symbol,''''
                    timeframe="1m",
                    since = int(start_time.timestamp() * 1000),
                    limit = 1000,
                
"
        if data:"""
        # Convert to DataFrame""""
                    df, pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])"
"""
        # Convert timestamp""""
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")"
"""
        # Save to parquet""""
                    df.to_parquet(file_path, compression="zstd", index = False)""
"""""
                    logger.info(f"✅ Downloaded {filename}: {len(df)} rows")"
        return True"""
                else:""""
                    logger.warning(f"⚠️ No data available for {month}")"
        return False"""
            else:""""
                logger.warning("⚠️ No exchange available for download")
        return False"
"""
        except Exception as e:""""
            logger.exception(f"❌ Error downloading {month}: {e}")
        return False"
"""
    async def _count_klines_rows(self, symbol: str, exchange: str) -> int:"""
        """Count total rows in klines files."""""
        try:"
            except Exception as e:"""
                pass""""
            pattern, f"klines_{exchange}_{symbol}_1m_*.parquet"
            files, list(self.data_cache_path.glob(pattern))

            total_rows, 0
        for file_path in files:
                df, pd.read_parquet(file_path)
                total_rows += len(df)
"
        return total_rows"""
        except Exception as e:""""
            logger.exception(f"❌ Error counting rows: {e}")"
        return 0""
"""""
    @with_tracing_span("download_futures_data")"
    @handle_errors()"""
        exceptions=(OSError, ValueError, TypeError, KeyError),""""
        default_return={"success": False, "error": "Download failed"},""""
        context="missing_data_downloader.download_futures_data"
    
    async def download_futures_data()"
        self, symbol: str, exchange: str, start_date: datetime, end_date: datetime"""
    ) -> dict:"""
        """"""Download futures data for a specific date range.""

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_date: Start date
            end_date: End date

        Returns:"
            Dictionary with download results""
"""
        """"""""""""""
        logger.info(f"📥 Downloading futures data for {exchange}_{symbol}")
"
        if not await self._ensure_exchange_initialized():"""
            pass""""
        return {"success": False, "error": "Exchange not initialized"}"
"""
        results = {}"""
            "success": True,"""
            "downloaded_months": 0,"""
            "failed_months": 0,"""
            "total_rows": 0,"""
            "errors"": [],"
        

        # Generate list of months to download
        current_date, start_date.replace(day = 1)
        months_to_download = []
"
        while current_date <= end_date:"""
        # Check if file already exists""""
            filename, f"futures_{exchange}_{symbol}_{current_date.strftime("%Y%m')}.parquet'
            file_path, self.data_cache_path / filename

        if not file_path.exists():'
                months_to_download.append(current_date)'''
            else:''''
                logger.debug(f"📁 File already exists: {filename}")

        # Move to next month
        if current_date.month == 12:
                current_date, current_date.replace(year = current_date.year + 1, month = 1)
            else:"
                current_date, current_date.replace(month = current_date.month + 1)""
"""""
        logger.info(f"📊 Found {len(months_to_download)} months to download")

        # Download data for each month
        for month in months_to_download:
            pass
        try:
            except Exception as e:
                pass"
                success, await self._download_single_futures_month(symbol, exchange, month)"""
        if success:""""
                    results["downloaded_months"] += 1"""
                else:""""
                    results["failed_months"] += 1""""
                    results["errors"].append(f"Failed to download {month}")

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)"
"""
        except Exception as e:""""
                results["failed_months"] += 1""""
                results["errors"].append(f"Error downloading {month}: {e}")""""
                logger.exception(f"❌ Error downloading {month}: {e}")"
"""
        # Count total rows""""
        results["total_rows"] = await self._count_futures_rows(symbol, exchange)"
"""
        logger.info()""""
            f"📊 Download complete: {results["downloaded_months']} downloaded, ''''''''
            f"{results["failed_months']} failed, {results['total_rows']} total rows'
        

        return results

    async def _download_single_futures_month()'
        self, symbol: str, exchange: str, month: datetime'''
    ) -> bool:''''
        """Download futures data for a single month."""
        try:
            except Exception as e:"
                pass"""
        # Create filename""""
            filename, f"futures_{exchange}_{symbol}_{month.strftime("%Y%m')}.parquet'
            file_path, self.data_cache_path / filename

        # Calculate month boundaries
            start_time, month.replace(day = 1)
        if month.month == 12:
                end_time, month.replace(year = month.year + 1, month = 1, day = 1) - timedelta(seconds = 1)
            else:
                end_time, month.replace(month = month.month + 1, day = 1) - timedelta(seconds = 1)

        # Download data using exchange API
        if self.exchange:
                data, await self.exchange.fetch_funding_rate()
                    symbol = symbol,
                    since = int(start_time.timestamp() * 1000),
                    limit = 1000,
                

        if data:
        # Convert to DataFrame
                    df, pd.DataFrame(data)'
'''
        # Ensure required columns''''
        if "timestamp" not in df.columns and "fundingTime" in df.columns:""""
                        df["timestamp"] = df["fundingTime"]"
"""
        # Convert timestamp""""
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")"
"""
        # Save to parquet""""
                    df.to_parquet(file_path, compression="zstd", index = False)""
"""""
                    logger.info(f"✅ Downloaded {filename}: {len(df)} rows")"
        return True"""
                else:""""
                    logger.warning(f"⚠️ No data available for {month}")"
        return False"""
            else:""""
                logger.warning("⚠️ No exchange available for download")
        return False"
"""
        except Exception as e:""""
            logger.exception(f"❌ Error downloading {month}: {e}")
        return False"
"""
    async def _count_futures_rows(self, symbol: str, exchange: str) -> int:"""
        """Count total rows in futures files."""""
        try:"
            except Exception as e:"""
                pass""""
            pattern, f"futures_{exchange}_{symbol}_*.parquet"
            files, list(self.data_cache_path.glob(pattern))

            total_rows, 0
        for file_path in files:
                df, pd.read_parquet(file_path)
                total_rows += len(df)
"
        return total_rows"""
        except Exception as e:""""
            logger.exception(f"❌ Error counting rows: {e}")"
        return 0""
"""""
    @with_tracing_span("download_all_missing_data")"
    @handle_errors()"""
        exceptions=(OSError, ValueError, TypeError, KeyError),""""
        default_return={"success": False, "error": "Download failed"},""""
        context="missing_data_downloader.download_all_missing_data"
    
    async def download_all_missing_data()"
        self, symbol: str, exchange: str, end_date: datetime | None, None"""
    ) -> dict:"""
        """"""Download all missing data for a symbol and exchange.""

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            end_date: End date for analysis (default: today)

        Returns:"
            Dictionary with download results""
"""
        """""""
        download_start, datetime.now()
"
        if end_date is None:"""
            end_date, datetime.now()""""
            logger.info(f"📅 No end_date provided, using default: {end_date.date()} (today)")
"
        start_date, end_date - timedelta(days = 365)  # Last year"""
""""
        logger.info(f"🚀 COMPREHENSIVE DATA DOWNLOAD FOR {exchange}_{symbol}")""""
        logger.info(f"📅 Download period: {start_date.date()} to {end_date.date()}")""""
        logger.info(f"📁 Data cache path: {self.data_cache_path}")""""
        logger.info("-" * 60)"
"""
        results = {}"""
            "success": True,"""
            "symbol": symbol,"""
            "exchange": exchange,"""
            "start_date": start_date,"""
            "end_date": end_date,"""
            "download_results": {},"""
            "errors"": [],"
        
"
        # Download aggtrades data"""
        aggtrades_results, await self.download_aggtrades_data(symbol, exchange, start_date, end_date)""""
        results["download_results"]["aggtrades"] = aggtrades_results""
"""""
        if not aggtrades_results["success"]:""""
            results["errors"].append("Aggtrades download failed")
"
        # Download klines data"""
        klines_results, await self.download_klines_data(symbol, exchange, start_date, end_date)""""
        results["download_results"]["klines"] = klines_results""
"""""
        if not klines_results["success"]:""""
            results["errors"].append("Klines download failed")
"
        # Download futures data"""
        futures_results, await self.download_futures_data(symbol, exchange, start_date, end_date)""""
        results["download_results"]["futures"] = futures_results""
"""""
        if not futures_results["success"]:""""
            results["errors"].append("Futures download failed")"
"""
        # Determine overall success""""
        if results["errors"]:""""
            results["success"] = False

        download_end, datetime.now()"
        download_time, download_end - download_start""
"""""
        logger.info("-" * 60)""""
        logger.info("📊 COMPREHENSIVE DOWNLOAD SUMMARY")""""
        logger.info(f"⏱️  Total download time: {download_time}")""""
        logger.info(f"🎯 Target: {exchange}_{symbol}")""""
        logger.info(f"📅 Period: {start_date.date()} to {end_date.date()}")""""
        logger.info(f"✅ Success: {results["success']}')''''
        logger.info(f"❌ Errors: {len(results["errors'])}')'
'''
        # Log individual download results''''
        for data_type, download_result in results["download_results"].items():"""
            pass""""
        if download_result.get("success"):""""
                logger.info(f"✅ {data_type.title()}: Downloaded successfully")"""
            else:""""
                logger.error(f"❌ {data_type.title()}: Download failed")""
"""""
        if results["errors"]:""""
            logger.error("❌ DOWNLOAD ERRORS:")""""
        for i, error in enumerate(results["errors"], 1):""""
                logger.error(f"  {i}. {error}")""
"""""
        if results["success"]:""""
            logger.info("🎉 COMPREHENSIVE DOWNLOAD COMPLETED SUCCESSFULLY!")"""
        else:""""
            logger.error("❌ COMPREHENSIVE DOWNLOAD COMPLETED WITH ERRORS!")"
"""
        return results"""''''''""""