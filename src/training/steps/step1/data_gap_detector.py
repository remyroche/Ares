"""Data Gap Detector.

Automatically detects missing days of aggtrades and months of klines and futures,
per exchange and per symbol in data_cache/.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    validate_data_completeness,
    validate_data_structure,
    with_tracing_span,
)
from src.utils.logger import system_logger

# Add project root to path
project_root, Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("DataGapDetector")


class DataGapDetector:
    """Detects missing data gaps in data_cache directory."""

    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

        # Import the gap filler for immediate gap filling
        try:
            from .missing_data_downloader_and_gap_filler import (
                MissingDataDownloaderAndGapFiller,
            )
        self.gap_filler = MissingDataDownloaderAndGapFiller(data_cache_path)
        except ImportError:
            logger.warning("⚠️ MissingDataDownloaderAndGapFiller not available - gap filling disabled")
        self.gap_filler = None

    @validate_data_structure
    @comprehensive_data_validation
    @with_tracing_span("detect_missing_data")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError)
        default_return={"symbol": "", "exchange": "", "start_date": None, "end_date": None
                       "missing_aggtrades_days": [], "missing_klines_months": [], "missing_futures_months": [],
                       "existing_aggtrades_days": [], "existing_klines_months": [], "existing_futures_months": []},
        context="data_gap_detector.detect_missing_data"
    )
    def detect_missing_data(self, symbol: str, exchange: str, start_date: datetime | None = None, end_date: datetime | None = None) -> dict:
        """Detect missing data for a specific symbol and exchange.

        Args:
            symbol: Trading symbol (e.g. = 'ETHUSDT')
            exchange: Exchange name (e.g. = 'BINANCE')
            start_date: Start date for analysis (default: 2 years ago)
            end_date: End date for analysis (default: today)

        Returns:
            Dictionary with missing data information

        """
        if start_date is None:
            start_date, datetime.now() - timedelta(days=365*2)
        if end_date is None:
            end_date = datetime.now()

        logger.info(f"🔍 Detecting missing data for {exchange}_{symbol} from {start_date.date()} to {end_date.date()}")

        results = {
            "symbol": symbol,
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "missing_aggtrades_days": [],
            "missing_klines_months": [],
            "missing_futures_months": [],
            "existing_aggtrades_days": [],
            "existing_klines_months": [],
            "existing_futures_months": [],
        }

        # Detect missing aggtrades (daily files)
        results.update(self._detect_missing_aggtrades(symbol, exchange, start_date, end_date))

        # Detect missing klines (monthly files)
        results.update(self._detect_missing_klines(symbol, exchange, start_date, end_date))

        # Detect missing futures (monthly files)
        results.update(self._detect_missing_futures(symbol, exchange, start_date, end_date))

        return results

    def _detect_missing_aggtrades(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> dict:
        """Detect missing aggtrades daily files."""
        # Get both CSV and Parquet files
        csv_pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        # Create a dictionary to track files by date, prioritizing Parquet over CSV
        files_by_date = {}

        # Add CSV files first
        for csv_file in csv_files:
        try:
        # Extract date from filename
                date_str, csv_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m%d").date()
                files_by_date[file_date] = csv_file
        except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same date)
        for parquet_file in parquet_files:
        try:
        # Extract date from filename
                date_str, parquet_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m%d").date()
                files_by_date[file_date] = parquet_file
        except (ValueError, IndexError):
                continue

        # Generate list of expected dates
        current_date = start_date.date()
        expected_dates = []
        while current_date <= end_date.date():
            expected_dates.append(current_date)
            current_date += timedelta(days=1)

        # Find missing and existing dates
        existing_dates = set(files_by_date.keys())
        missing_dates = [date for date in expected_dates if date not in existing_dates]

        return {
            "missing_aggtrades_days": missing_dates,
            "existing_aggtrades_days": list(existing_dates),
        }

    def _detect_missing_klines(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> dict:
        """Detect missing klines monthly files."""
        # Get both CSV and Parquet files
        csv_pattern = f"klines_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"klines_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        # Create a dictionary to track files by month
        files_by_month = {}

        # Add CSV files first
        for csv_file in csv_files:
        try:
        # Extract date from filename
                date_str, csv_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = csv_file
        except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same month)
        for parquet_file in parquet_files:
        try:
        # Extract date from filename
                date_str, parquet_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = parquet_file
        except (ValueError, IndexError):
                continue

        # Generate list of expected months
        current_date, start_date.replace(day=1)
        expected_months = []
        while current_date <= end_date:
            expected_months.append(current_date.date())
        # Move to next month
        if current_date.month == 12:
                current_date, current_date.replace(year=current_date.year + 1, month=1)
            else: current_date = current_date.replace(month=current_date.month + 1)

        # Find missing and existing months
        existing_months = set(files_by_month.keys())
        missing_months = [month for month in expected_months if month not in existing_months]

        return {
            "missing_klines_months": missing_months,
            "existing_klines_months": list(existing_months),
        }

    def _detect_missing_futures(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> dict:
        """Detect missing futures monthly files."""
        # Get both CSV and Parquet files
        csv_pattern = f"futures_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"futures_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        # Create a dictionary to track files by month
        files_by_month = {}

        # Add CSV files first
        for csv_file in csv_files:
        try:
        # Extract date from filename
                date_str, csv_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = csv_file
        except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same month)
        for parquet_file in parquet_files:
        try:
        # Extract date from filename
                date_str, parquet_file.stem.split("_")[-1]
                file_date, datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = parquet_file
        except (ValueError, IndexError):
                continue

        # Generate list of expected months
        current_date, start_date.replace(day=1)
        expected_months = []
        while current_date <= end_date:
            expected_months.append(current_date.date())
        # Move to next month
        if current_date.month == 12:
                current_date, current_date.replace(year=current_date.year + 1, month=1)
            else: current_date = current_date.replace(month=current_date.month + 1)

        # Find missing and existing months
        existing_months = set(files_by_month.keys())
        missing_months = [month for month in expected_months if month not in existing_months]

        return {
            "missing_futures_months": missing_months,
            "existing_futures_months": list(existing_months),
        }

    @validate_data_completeness
    @with_tracing_span("detect_aggtrades_gaps")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError
                   FileNotFoundError = PermissionError, pd.errors.ParserError),
        default_return=[]
        context="data_gap_detector.detect_aggtrades_gaps"
    )
    def detect_aggtrades_gaps(self, symbol: str, exchange: str, min_gap_seconds: int = 10) -> list[dict]:
        """Detect gaps over specified seconds in aggtrades files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            min_gap_seconds: Minimum gap size to report (default: 10 seconds)

        Returns:
            List of gap information dictionaries

        """
        # Get both CSV and Parquet files
        csv_pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        logger.info(f"📊 Found {len(csv_files)} CSV files and {len(parquet_files)} Parquet files")

        # Create a dictionary to track files by date, prioritizing Parquet over CSV
        files_by_date = {}

        # Add CSV files first
        for csv_file in csv_files:
        try: date_str = csv_file.stem.split("_")[-1]
                files_by_date[date_str] = csv_file
        except (ValueError, IndexError):
                continue

        # Add Parquet files, overwriting CSV files for the same date
        for parquet_file in parquet_files:
        try: date_str = parquet_file.stem.split("_")[-1]
                files_by_date[date_str] = parquet_file
        except (ValueError, IndexError):
                continue

        aggtrades_files = list(files_by_date.values())
        logger.info(f"📊 Processing {len(aggtrades_files)} unique date files (Parquet prioritized over CSV)")

        gaps = []
        files_processed = 0
        files_with_gaps = 0

        for file_path in aggtrades_files:
        try:
                files_processed += 1

        # Only log every 25 files to reduce noise
        if files_processed % 25 == 0:
                    logger.info(f"🔍 Processing file {files_processed}/{len(aggtrades_files)}: {file_path.name}")

        # Read the file based on its format
        if file_path.suffix.lower() == ".csv":
                    df, pd.read_csv(file_path, parse_dates=["timestamp"])
                elif file_path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    logger.warning(f"⚠️ Skipping unsupported file format: {file_path.name}")
                    continue

        if len(df) < 2:
                    continue

        # Sort by timestamp
                df = df.sort_values("timestamp")

        # Calculate time differences
                time_diffs = df["timestamp"].diff().dropna()

        # Find gaps larger than min_gap_seconds
                large_gaps, time_diffs[time_diffs > pd.Timedelta(seconds=min_gap_seconds)]

                file_has_gaps = False
        for idx, gap in large_gaps.items():
                    gap_start = df.loc[idx-1, "timestamp"]
                    gap_end = df.loc[idx, "timestamp"]
                    gap_duration = gap.total_seconds()

                    gap_info = {
                        "file": file_path.name,
                        "gap_start": gap_start,
                        "gap_end": gap_end,
                        "gap_duration_seconds": gap_duration,
                        "gap_duration_formatted": str(gap),
                    }

                    gaps.append(gap_info)
                    file_has_gaps = True

        if file_has_gaps:
                    files_with_gaps += 1
                    logger.warning(f"🚨 GAPS FOUND in {file_path.name}: {len(large_gaps)} gaps detected")

        except Exception as e:
                logger.exception(f"❌ Error processing {file_path.name}: {e}")
                continue

        # Generate comprehensive gap summary
        logger.info("=" * 80)
        logger.info("🚨 AGGTRADES GAP DETECTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Files processed: {files_processed}")
        logger.info(f"📊 Files with gaps: {files_with_gaps}")
        logger.info(f"📊 Total gaps found: {len(gaps)}")
        logger.info(f"📊 Minimum gap threshold: {min_gap_seconds} seconds")

        if gaps:
            logger.warning("🚨 GAPS DETECTED - DETAILED BREAKDOWN:")
            logger.warning("=" * 60)

        # Group gaps by duration
            gap_durations = [gap["gap_duration_seconds"] for gap in gaps]
            max_gap = max(gap_durations)
            min_gap = min(gap_durations)
            avg_gap = sum(gap_durations) / len(gap_durations)

            logger.warning("📊 Gap Statistics:")
            logger.warning(f"   • Largest gap: {max_gap:.1f} seconds ({max_gap/60:.1f} minutes)")
            logger.warning(f"   • Smallest gap: {min_gap:.1f} seconds")
            logger.warning(f"   • Average gap: {avg_gap:.1f} seconds")

        # Show top 5 largest gaps
            sorted_gaps, sorted(gaps, key=lambda x: x["gap_duration_seconds"], reverse=True)
            logger.warning("🚨 TOP 5 LARGEST GAPS:")
        for i, gap in enumerate(sorted_gaps[:5], 1):
                logger.warning(f"   {i}. {gap['file']}: {gap['gap_start']} to {gap['gap_end']} ({gap['gap_duration_seconds']:.1f}s)")

        # Show files with most gaps
            files_gap_count = {}
        for gap in gaps:
                file_name = gap["file"]
                files_gap_count[file_name] = files_gap_count.get(file_name, 0) + 1

            top_files, sorted(files_gap_count.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.warning("🚨 FILES WITH MOST GAPS:")
        for file_name, gap_count in top_files:
                logger.warning(f"   • {file_name}: {gap_count} gaps")
        else:
            logger.info("✅ No gaps detected - data quality is good!")

        logger.info("=" * 80)
        return gaps

    @validate_data_completeness
    @with_tracing_span("detect_and_fill_aggtrades_gaps")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, pd.errors.EmptyDataError
                   FileNotFoundError = PermissionError, pd.errors.ParserError),
        default_return={"files_processed": 0, "files_with_gaps": 0, "total_gaps": 0, "gaps_filled": 0, "gaps_failed": 0}
        context="data_gap_detector.detect_and_fill_aggtrades_gaps"
    )
    async def detect_and_fill_aggtrades_gaps(self, symbol: str, exchange: str, min_gap_seconds: int = 10, auto_fill: bool = True) -> dict:
        """Detect gaps over specified seconds in aggtrades files and fill them immediately.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            min_gap_seconds: Minimum gap size to report (default: 10 seconds)
            auto_fill: Whether to automatically fill gaps when found

        Returns:
            Dictionary with gap detection and filling results

        """
        if not self.gap_filler and auto_fill:
            logger.warning("⚠️ Gap filler not available - running in detection-only mode")
            auto_fill = False

        # Get both CSV and Parquet files
        csv_pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        logger.info(f"📊 Found {len(csv_files)} CSV files and {len(parquet_files)} Parquet files")

        # Create a dictionary to track files by date, prioritizing Parquet over CSV
        files_by_date = {}

        # Add CSV files first
        for csv_file in csv_files:
        try: date_str = csv_file.stem.split("_")[-1]
                files_by_date[date_str] = csv_file
        except (ValueError, IndexError):
                continue

        # Add Parquet files, overwriting CSV files for the same date
        for parquet_file in parquet_files:
        try: date_str = parquet_file.stem.split("_")[-1]
                files_by_date[date_str] = parquet_file
        except (ValueError, IndexError):
                continue

        aggtrades_files = list(files_by_date.values())
        logger.info(f"📊 Processing {len(aggtrades_files)} unique date files (Parquet prioritized over CSV)")

        results = {
            "files_processed": 0,
            "files_with_gaps": 0,
            "total_gaps": 0,
            "gaps_filled": 0,
            "gaps_failed": 0,
            "gaps_by_file": {},
        }

        for file_path in aggtrades_files:
        try:
                results["files_processed"] += 1

        # Only log every 25 files to reduce noise
        if results["files_processed"] % 25 == 0:
                    logger.info(f"🔍 Processing file {results['files_processed']}/{len(aggtrades_files)}: {file_path.name}")

        # Read the file based on its format
        if file_path.suffix.lower() == ".csv":
                    df, pd.read_csv(file_path, parse_dates=["timestamp"])
                elif file_path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    logger.warning(f"⚠️ Skipping unsupported file format: {file_path.name}")
                    continue

        if len(df) < 2:
                    continue

        # Sort by timestamp
                df = df.sort_values("timestamp")

        # Calculate time differences
                time_diffs = df["timestamp"].diff().dropna()

        # Find gaps larger than min_gap_seconds
                large_gaps, time_diffs[time_diffs > pd.Timedelta(seconds=min_gap_seconds)]

        if len(large_gaps) > 0:
                    results["files_with_gaps"] += 1
                    results["total_gaps"] += len(large_gaps)

                    logger.warning(f"🚨 GAPS FOUND in {file_path.name}: {len(large_gaps)} gaps detected")

        # Prepare gaps for this file
                    file_gaps = []
        for idx, gap in large_gaps.items():
                        gap_start = df.loc[idx-1, "timestamp"]
                        gap_end = df.loc[idx, "timestamp"]
                        gap_duration = gap.total_seconds()

                        gap_info = {
                            "file": file_path.name,
                            "gap_start": gap_start,
                            "gap_end": gap_end,
                            "gap_duration_seconds": gap_duration,
                            "gap_duration_formatted": str(gap),
                        }

                        file_gaps.append(gap_info)

        # Fill gap immediately if auto_fill is enabled
        if auto_fill and self.gap_filler:
        try:
                                logger.info(f"🔧 Filling gap in {file_path.name}: {gap_start} to {gap_end} ({gap_duration:.1f}s)")

        # Fill this specific gap
                                fill_result, await self.gap_filler.fill_single_gap(
                                    symbol=symbol
                                    exchange=exchange
                                    gap_info=gap_info
                                )

        if fill_result.get("success", False):
                                    results["gaps_filled"] += 1
                                    logger.info(f"✅ Gap filled successfully: {gap_duration:.1f}s")
                                else:
                                    results["gaps_failed"] += 1
                                    logger.error(f"❌ Failed to fill gap: {fill_result.get('error', 'Unknown error')}")

        except Exception as e:
                                results["gaps_failed"] += 1
                                logger.exception(f"❌ Error filling gap in {file_path.name}: {e}")

                    results["gaps_by_file"][file_path.name] = file_gaps

        except Exception as e:
                logger.exception(f"❌ Error processing {file_path.name}: {e}")
                continue

        # Generate comprehensive summary
        logger.info("=" * 80)
        logger.info("🚨 AGGTRADES GAP DETECTION AND FILLING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Files processed: {results['files_processed']}")
        logger.info(f"📊 Files with gaps: {results['files_with_gaps']}")
        logger.info(f"📊 Total gaps found: {results['total_gaps']}")
        logger.info(f"📊 Gaps filled: {results['gaps_filled']}")
        logger.info(f"📊 Gaps failed: {results['gaps_failed']}")
        logger.info(f"📊 Minimum gap threshold: {min_gap_seconds} seconds")

        if results["total_gaps"] > 0:
            logger.warning("🚨 GAPS PROCESSING SUMMARY:")
            logger.warning("=" * 60)

        # Show files with most gaps
            files_gap_count = {file: len(gaps) for file, gaps in results["gaps_by_file"].items()}
            top_files, sorted(files_gap_count.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.warning("🚨 FILES WITH MOST GAPS:")
        for file_name, gap_count in top_files:
                logger.warning(f"   • {file_name}: {gap_count} gaps")

        if auto_fill:
                success_rate = (results["gaps_filled"] / results["total_gaps"]) * 100 if results["total_gaps"] > 0 else 0
                logger.info(f"📊 Gap filling success rate: {success_rate:.1f}%")
        else:
            logger.info("✅ No gaps detected - data quality is good!")

        logger.info("=" * 80)
        return results

    def generate_missing_data_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive report of missing data."""
        results, self.detect_missing_data(symbol, exchange)
        gaps, self.detect_aggtrades_gaps(symbol, exchange)

        return f"""
🔍 MISSING DATA REPORT FOR {exchange}_{symbol}
{'='*60}

📊 MISSING DATA SUMMARY:
    pass
• Missing Aggtrades Days: {len(results['missing_aggtrades_days'])}
• Missing Klines Months: {len(results['missing_klines_months'])}
• Missing Futures Months: {len(results['missing_futures_months'])}

📈 DATA GAPS:
    pass
• Gaps > 10 seconds: {len(gaps)}

📅 MISSING AGGTRADES DAYS:
    pass
{chr(10).join(f'  • {date}' for date in results['missing_aggtrades_days'][:10])}
{'  ...' if len(results['missing_aggtrades_days']) > 10 else ''}

📊 MISSING KLINES MONTHS:
    pass
{chr(10).join(f'  • {date}' for date in results['missing_klines_months'])}

📈 MISSING FUTURES MONTHS:
    pass
{chr(10).join(f'  • {date}' for date in results['missing_futures_months'])}

⚠️ DATA GAPS (>10s):
    pass
{chr(10).join(f'  • {gap["file"]}: {gap["gap_start"]} to {gap["gap_end"]} ({gap["gap_duration_seconds"]:.1f}s)' for gap in gaps[:5])}
{'  ...' if len(gaps) > 5 else ''}
"""