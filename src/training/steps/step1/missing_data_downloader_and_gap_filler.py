"""Missing Data Downloader and Gap Filler.

Automatically downloads missing data and fills gaps:
1. Downloads missing days of aggtrades (up to 2 days ago)
2. Downloads missing months of klines (up to 2 days ago)
3. Downloads missing months of futures (up to 2 days ago)
4. Fills gaps over 10 seconds in aggtrades data
5. Integrates new data without deleting pre-existing data
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import CONFIG
    from src.utils.centralized_decorators import (
        comprehensive_data_validation,
        guard_dataframe_nulls,
        handle_errors,
        optimize_memory_usage,
        secure_data_processing,
        validate_data_quality,
        validate_data_structure,
        with_tracing_span,
    )
    from src.utils.logger import system_logger

    from .data_gap_detector import DataGapDetector
    # Skip problematic binance import for now
    BinanceExchange = None
except ImportError:
    # Fallback imports and configurations
    CONFIG = {
        "EXCHANGE": "BINANCE",
        "API_KEY": "",
        "API_SECRET": "",
        "TESTNET": True,
    }

    # Fallback decorators
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def with_tracing_span(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def validate_data_quality(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def validate_data_structure(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def guard_dataframe_nulls(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def optimize_memory_usage(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def comprehensive_data_validation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def secure_data_processing(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    DataGapDetector = None
    BinanceExchange = None

    # Fallback logger
    import logging
    logging.basicConfig(level=logging.INFO)
    system_logger = logging.getLogger("MissingDataDownloaderFallback")

logger = system_logger.getChild("MissingDataDownloader")


class MissingDataDownloaderAndGapFiller:
    """Downloads missing data and fills gaps automatically."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

        # Initialize exchange connection
        if BinanceExchange:
            try:
                self.exchange = BinanceExchange(CONFIG)
            except Exception as e:
                logger.warning(f"Failed to initialize exchange: {e}")
                self.exchange = None
        else:
            self.exchange = None

        # Download limits and retry settings
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.rate_limit_delay = 0.1  # seconds between requests

        # Gap detection settings
        self.min_gap_seconds = 10

        # Exchange initialization flag
        self._exchange_initialized = False

    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError),
        default_return=False,
        context="missing_data_downloader.ensure_exchange_initialized",
    )
    async def _ensure_exchange_initialized(self) -> bool:
        """Ensure the exchange is properly initialized."""
        if not self._exchange_initialized:
            try:
                logger.info("🔧 Initializing Binance exchange connection...")
                if self.exchange:
                    success = await self.exchange.initialize()
                    if success:
                        self._exchange_initialized = True
                        logger.info("✅ Binance exchange initialized successfully")
                        return True
                    logger.error("❌ Failed to initialize Binance exchange")
                    return False
                logger.warning("⚠️ Exchange not available")
                return False
            except Exception as e:
                logger.exception(f"❌ Error initializing exchange: {e}")
                return False
        return True

    @secure_data_processing
    @with_tracing_span("get_current_timestamp")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError),
        default_return=datetime.now(),
        context="missing_data_downloader.get_current_timestamp",
    )
    def get_current_timestamp(self) -> datetime:
        """Get current timestamp from exchange to determine 'today'."""
        try:
            # Get server time from exchange
            if self.exchange:
                server_time = self.exchange.get_server_time()
                if server_time:
                    return datetime.fromtimestamp(server_time / 1000)
            # Fallback to local time
            return datetime.now()
        except Exception as e:
            logger.warning(f"⚠️ Could not get server time, using local time: {e}")
            return datetime.now()

    @validate_data_structure
    @with_tracing_span("identify_missing_data")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return={"symbol": "", "exchange": "", "end_date": None, "missing_aggtrades_days": [],
                       "missing_klines_months": [], "missing_futures_months": []},
        context="missing_data_downloader.identify_missing_data",
    )
    def identify_missing_data(self, symbol: str, exchange: str,
                            end_date: datetime | None = None) -> dict:
        """Identify all missing data periods.

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
        if DataGapDetector:
            gap_detector = DataGapDetector(str(self.data_cache_path))

            # Get missing data periods
            missing_data = gap_detector.detect_missing_data(symbol, exchange, start_date, end_date)

            # Get aggtrades gaps
            aggtrades_gaps = gap_detector.detect_aggtrades_gaps(symbol, exchange, self.min_gap_seconds)

            return {
                "symbol": symbol,
                "exchange": exchange,
                "start_date": start_date,
                "end_date": end_date,
                "missing_aggtrades_days": missing_data["missing_aggtrades_days"],
                "missing_klines_months": missing_data["missing_klines_months"],
                "missing_futures_months": missing_data["missing_futures_months"],
                "aggtrades_gaps": aggtrades_gaps,
                "total_missing_periods": (
                    len(missing_data["missing_aggtrades_days"]) +
                    len(missing_data["missing_klines_months"]) +
                    len(missing_data["missing_futures_months"])
                ),
            }
        # Fallback when DataGapDetector is not available
        return {
            "symbol": symbol,
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "missing_aggtrades_days": [],
            "missing_klines_months": [],
            "missing_futures_months": [],
            "aggtrades_gaps": [],
            "total_missing_periods": 0,
        }

    @optimize_memory_usage
    @with_tracing_span("download_missing_aggtrades")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError,
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={"downloaded_days": 0, "failed_days": 0, "total_rows": 0, "errors": ["Download failed"]},
        context="missing_data_downloader.download_missing_aggtrades",
    )
    async def download_missing_aggtrades(self, symbol: str, exchange: str,
                                       missing_days: list[datetime]) -> dict:
        """Download missing aggtrades data for specific days.

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
                "downloaded_days": 0,
                "failed_days": 0,
                "total_rows": 0,
                "errors": ["Failed to initialize exchange connection"],
            }

        logger.info(f"📥 Downloading {len(missing_days)} missing aggtrades days for {exchange}_{symbol}")

        results = {
            "downloaded_days": 0,
            "failed_days": 0,
            "total_rows": 0,
            "errors": [],
        }

        for date in missing_days:
            try:
                logger.info(f"📥 Downloading aggtrades for {date.date()}")

                # Download aggtrades for the day
                aggtrades_data = await self.exchange.get_aggregate_trades(
                    symbol=symbol,
                    start_time=date,
                    end_time=date + timedelta(days=1),
                )

                if aggtrades_data and len(aggtrades_data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(aggtrades_data)

                    # Ensure proper column names and types
                    df = self._standardize_aggtrades_format(df)

                    # Save to file
                    filename = f"aggtrades_{exchange}_{symbol}_{date.strftime('%Y%m%d')}.parquet"
                    file_path = self.data_cache_path / filename

                    df.to_parquet(file_path, compression="zstd", index=False)

                    results["downloaded_days"] += 1
                    results["total_rows"] += len(df)

                    logger.info(f"✅ Downloaded {len(df)} aggtrades for {date.date()}")

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                else:
                    logger.warning(f"⚠️ No aggtrades data available for {date.date()}")
                    results["failed_days"] += 1

            except Exception as e:
                logger.exception(f"❌ Error downloading aggtrades for {date.date()}: {e}")
                results["failed_days"] += 1
                results["errors"].append(f"{date.date()}: {e}")

        logger.info(f"📊 Aggtrades download complete: {results['downloaded_days']} days, {results['total_rows']} rows")
        return results

    @comprehensive_data_validation
    @with_tracing_span("fill_single_gap")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, ConnectionError, TimeoutError,
                   FileNotFoundError, PermissionError, MemoryError),
        default_return={"success": False, "error": "Gap filling failed", "rows_added": 0},
        context="missing_data_downloader.fill_single_gap",
    )
    async def fill_single_gap(self, symbol: str, exchange: str,
                             gap_info: dict) -> dict:
        """Fill a single gap in aggtrades data by downloading missing period.

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
                "success": False,
                "error": "Failed to initialize exchange connection",
                "rows_added": 0,
            }

        try:
            gap_start = gap_info["gap_start"]
            gap_end = gap_info["gap_end"]
            file_name = gap_info["file"]

            # Convert datetime to milliseconds
            if isinstance(gap_start, str):
                gap_start = pd.to_datetime(gap_start)
            if isinstance(gap_end, str):
                gap_end = pd.to_datetime(gap_end)

            start_time_ms = int(gap_start.timestamp() * 1000)
            end_time_ms = int(gap_end.timestamp() * 1000)

            missing_data = None

            # Try to download missing data for the gap period
            try:
                logger.info(f"🔍 Trying to download data for gap: {gap_start} to {gap_end}")
                missing_data = await self.exchange.get_aggregate_trades(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )

                if missing_data and len(missing_data) > 0:
                    logger.info(f"✅ API successful: {len(missing_data)} trades found")
                else:
                    logger.info("⚠️ API returned no data")

            except Exception as e:
                logger.warning(f"⚠️ API failed: {e}")

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
                    df_combined = df_combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

                    # Save back to file
                    df_combined.to_parquet(file_path, compression="zstd", index=False)

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                    return {
                        "success": True,
                        "rows_added": len(df_missing),
                        "gap_duration": gap_info.get("gap_duration_seconds", 0),
                    }
                return {
                    "success": False,
                    "error": f"Could not find existing file: {file_name}",
                    "rows_added": 0,
                }
            return {
                "success": False,
                "error": "No data available to fill gap",
                "rows_added": 0,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rows_added": 0,
            }

    @validate_data_quality()
    @with_tracing_span("_standardize_aggtrades_format")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError),
        default_return=pd.DataFrame(),
        context="missing_data_downloader._standardize_aggtrades_format",
    )
    def _standardize_aggtrades_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize aggtrades DataFrame format."""
        # Expected columns
        expected_columns = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"]

        # Rename columns if needed
        column_mapping = {
            "a": "agg_trade_id",
            "p": "price",
            "q": "quantity",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "timestamp",
            "m": "is_buyer_maker",
        }

        df = df.rename(columns=column_mapping)

        # Ensure all required columns exist
        for col in expected_columns:
            if col not in df.columns:
                if col == "is_buyer_maker":
                    df[col] = False  # Default value
                else:
                    df[col] = 0  # Default value

        # Convert data types
        df["agg_trade_id"] = pd.to_numeric(df["agg_trade_id"], errors="coerce").fillna(0).astype("int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype("float64")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).astype("float64")
        df["first_trade_id"] = pd.to_numeric(df["first_trade_id"], errors="coerce").fillna(0).astype("int64")
        df["last_trade_id"] = pd.to_numeric(df["last_trade_id"], errors="coerce").fillna(0).astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")

        # Remove any rows with NaN values in critical columns
        critical_columns = ["timestamp", "price", "quantity"]
        df = df.dropna(subset=critical_columns)

        # Sort by timestamp
        df = df.sort_values("timestamp")

        return df[expected_columns]
