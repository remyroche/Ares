#!/usr/bin/env python3
"""Data Gap Detector for Step1.

Detects missing data gaps in aggtrades, klines, and futures files.
"""

from src.utils.centralized_decorators import (
    comprehensive_data_validation,
    handle_errors,
    validate_data_structure,
    with_tracing_span,
)
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


logger = system_logger.getChild("DataGapDetector")


class DataGapDetector:
    """Detects missing data gaps in trading data files."""

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
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return={
            "symbol": "",
            "exchange": "",
            "start_date": None,
            "end_date": None,
            "missing_aggtrades_days": [],
            "missing_klines_months": [],
            "missing_futures_months": [],
            "existing_aggtrades_days": [],
            "existing_klines_months": [],
            "existing_futures_months": [],
        },
        context="data_gap_detector.detect_missing_data",
    )
    def detect_missing_data(
        self, symbol: str, exchange: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict:
        """Detect missing data for a specific symbol and exchange.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            start_date: Start date for analysis (default: 2 years ago)
            end_date: End date for analysis (default: today)

        Returns:
            Dictionary with missing data information

        """
        detection_start = datetime.now()

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 2)
            logger.info(f"📅 No start_date provided, using default: {start_date.date()} (2 years ago)")
        if end_date is None:
            end_date = datetime.now()
            logger.info(f"📅 No end_date provided, using default: {end_date.date()} (today)")

        logger.info(f"🔍 DETECTING MISSING DATA FOR {exchange}_{symbol}")
        logger.info(f"📅 Analysis period: {start_date.date()} to {end_date.date()}")
        logger.info(f"📁 Data cache path: {self.data_cache_path}")
        logger.info("-" * 60)

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
        logger.info("📊 DETECTING MISSING AGGTRADES (DAILY FILES)")
        aggtrades_results = self._detect_missing_aggtrades(symbol, exchange, start_date, end_date)
        results.update(aggtrades_results)
        logger.info(
            f"📈 Aggtrades: {len(aggtrades_results['existing_aggtrades_days'])} existing, {len(aggtrades_results['missing_aggtrades_days'])} missing"
        )

        # Detect missing klines (monthly files)
        logger.info("📊 DETECTING MISSING KLINES (MONTHLY FILES)")
        klines_results = self._detect_missing_klines(symbol, exchange, start_date, end_date)
        results.update(klines_results)
        logger.info(
            f"📈 Klines: {len(klines_results['existing_klines_months'])} existing, {len(klines_results['missing_klines_months'])} missing"
        )

        # Detect missing futures (monthly files)
        logger.info("📊 DETECTING MISSING FUTURES (MONTHLY FILES)")
        futures_results = self._detect_missing_futures(symbol, exchange, start_date, end_date)
        results.update(futures_results)
        logger.info(
            f"📈 Futures: {len(futures_results['existing_futures_months'])} existing, {len(futures_results['missing_futures_months'])} missing"
        )

        # Summary
        total_missing = (
            len(results["missing_aggtrades_days"])
            + len(results["missing_klines_months"])
            + len(results["missing_futures_months"])
        )
        total_existing = (
            len(results["existing_aggtrades_days"])
            + len(results["existing_klines_months"])
            + len(results["existing_futures_months"])
        )

        detection_end = datetime.now()
        detection_time = detection_end - detection_start

        logger.info("-" * 60)
        logger.info("📊 MISSING DATA DETECTION SUMMARY")
        logger.info(f"⏱️  Detection time: {detection_time}")
        logger.info(f"📈 Total existing files: {total_existing}")
        logger.info(f"❌ Total missing files: {total_missing}")
        logger.info(
            f"📊 Coverage: {total_existing / (total_existing + total_missing) * 100:.1f}%"
            if (total_existing + total_missing) > 0
            else "📊 Coverage: N/A"
        )

        if total_missing > 0:
            logger.warning(f"⚠️  {total_missing} missing data files detected!")
        else:
            logger.info("✅ All expected data files are present!")

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
                date_str = csv_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
                files_by_date[file_date] = csv_file
            except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same date)
        for parquet_file in parquet_files:
            try:
                # Extract date from filename
                date_str = parquet_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
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
        existing_dates = list(files_by_date.keys())
        missing_dates = [date for date in expected_dates if date not in existing_dates]

        return {
            "existing_aggtrades_days": sorted(existing_dates),
            "missing_aggtrades_days": sorted(missing_dates),
        }

    def _detect_missing_klines(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> dict:
        """Detect missing klines monthly files."""
        # Get both CSV and Parquet files
        csv_pattern = f"klines_{exchange}_{symbol}_1m_*.csv"
        parquet_pattern = f"klines_{exchange}_{symbol}_1m_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        # Create a dictionary to track files by month, prioritizing Parquet over CSV
        files_by_month = {}

        # Add CSV files first
        for csv_file in csv_files:
            try:
                # Extract date from filename
                date_str = csv_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = csv_file
            except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same month)
        for parquet_file in parquet_files:
            try:
                # Extract date from filename
                date_str = parquet_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = parquet_file
            except (ValueError, IndexError):
                continue

        # Generate list of expected months
        current_date = start_date.replace(day=1).date()
        expected_months = []
        while current_date <= end_date.date():
            expected_months.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        # Find missing and existing months
        existing_months = list(files_by_month.keys())
        missing_months = [month for month in expected_months if month not in existing_months]

        return {
            "existing_klines_months": sorted(existing_months),
            "missing_klines_months": sorted(missing_months),
        }

    def _detect_missing_futures(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> dict:
        """Detect missing futures monthly files."""
        # Get both CSV and Parquet files
        csv_pattern = f"futures_{exchange}_{symbol}_*.csv"
        parquet_pattern = f"futures_{exchange}_{symbol}_*.parquet"

        csv_files = list(self.data_cache_path.glob(csv_pattern))
        parquet_files = list(self.data_cache_path.glob(parquet_pattern))

        # Create a dictionary to track files by month, prioritizing Parquet over CSV
        files_by_month = {}

        # Add CSV files first
        for csv_file in csv_files:
            try:
                # Extract date from filename
                date_str = csv_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = csv_file
            except (ValueError, IndexError):
                continue

        # Add Parquet files (overwrite CSV if same month)
        for parquet_file in parquet_files:
            try:
                # Extract date from filename
                date_str = parquet_file.stem.split("_")[-1]
                file_date = datetime.strptime(date_str, "%Y%m").date()
                files_by_month[file_date] = parquet_file
            except (ValueError, IndexError):
                continue

        # Generate list of expected months
        current_date = start_date.replace(day=1).date()
        expected_months = []
        while current_date <= end_date.date():
            expected_months.append(current_date)
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        # Find missing and existing months
        existing_months = list(files_by_month.keys())
        missing_months = [month for month in expected_months if month not in existing_months]

        return {
            "existing_futures_months": sorted(existing_months),
            "missing_futures_months": sorted(missing_months),
        }

    @with_tracing_span("detect_aggtrades_gaps")
    @handle_errors(
        exceptions=(OSError, ValueError, TypeError, KeyError, FileNotFoundError, PermissionError),
        default_return=[],
        context="data_gap_detector.detect_aggtrades_gaps",
    )
    def detect_aggtrades_gaps(self, symbol: str, exchange: str, min_gap_seconds: int = 10) -> list[dict]:
        """Detect gaps within aggtrades files.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            min_gap_seconds: Minimum gap duration to report (default: 10 seconds)

        Returns:
            List of gap information dictionaries

        """
        logger.info(f"🔍 Detecting aggtrades gaps for {exchange}_{symbol}")

        # Get all aggtrades files
        pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))
        pattern_parquet = f"aggtrades_{exchange}_{symbol}_*.parquet"
        parquet_files = list(self.data_cache_path.glob(pattern_parquet))

        all_files = sorted(csv_files + parquet_files)
        gaps = []

        for file_path in all_files:
            try:
                # Read the file
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                else:
                    df = pd.read_parquet(file_path)

                if len(df) < 2:
                    continue

                # Sort by timestamp
                df = df.sort_values("timestamp").reset_index(drop=True)

                # Calculate time differences
                df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

                # Find gaps larger than threshold
                gap_rows = df[df["time_diff"] > min_gap_seconds]

                for idx, row in gap_rows.iterrows():
                    if idx > 0:
                        gap_start = df.loc[idx - 1, "timestamp"]
                        gap_end = row["timestamp"]
                        gap_duration = (gap_end - gap_start).total_seconds()

                        gaps.append(
                            {
                                "file": file_path.name,
                                "gap_start": gap_start,
                                "gap_end": gap_end,
                                "gap_duration_seconds": gap_duration,
                                "data_type": "aggtrades",
                            }
                        )

            except Exception as e:
                logger.exception(f"❌ Error processing {file_path.name}: {e}")
                continue

        logger.info(f"📊 Found {len(gaps)} aggtrades gaps")
        return gaps

    @with_tracing_span("generate_missing_data_report")
    def generate_missing_data_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive missing data report.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Report string

        """
        # Detect missing data
        missing_data = self.detect_missing_data(symbol, exchange)

        report = f"""
🔍 MISSING DATA REPORT FOR {exchange}_{symbol}
{'=' * 60}

📅 ANALYSIS PERIOD:
• Start Date: {missing_data['start_date'].date()}
• End Date: {missing_data['end_date'].date()}

📊 AGGTRADES DATA:
• Existing Days: {len(missing_data['existing_aggtrades_days'])}
• Missing Days: {len(missing_data['missing_aggtrades_days'])}

📊 KLINES DATA:
• Existing Months: {len(missing_data['existing_klines_months'])}
• Missing Months: {len(missing_data['missing_klines_months'])}

📊 FUTURES DATA:
• Existing Months: {len(missing_data['existing_futures_months'])}
• Missing Months: {len(missing_data['missing_futures_months'])}

📋 MISSING AGGTRADES DAYS (first 20):
"""

        for date in missing_data["missing_aggtrades_days"][:20]:
            report += f"• {date}\n"

        if len(missing_data["missing_aggtrades_days"]) > 20:
            report += f"... and {len(missing_data['missing_aggtrades_days']) - 20} more days\n"

        report += f"""
📋 MISSING KLINES MONTHS (first 10):
"""

        for month in missing_data["missing_klines_months"][:10]:
            report += f"• {month}\n"

        if len(missing_data["missing_klines_months"]) > 10:
            report += f"... and {len(missing_data['missing_klines_months']) - 10} more months\n"

        report += f"""
📋 MISSING FUTURES MONTHS (first 10):
"""

        for month in missing_data["missing_futures_months"][:10]:
            report += f"• {month}\n"

        if len(missing_data["missing_futures_months"]) > 10:
            report += f"... and {len(missing_data['missing_futures_months']) - 10} more months\n"

        report += f"""
{'=' * 60}
"""

        return report
