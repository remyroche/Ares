"""
Enhanced Missing Value Handler

This module provides sophisticated missing value handling including:
- Forward fill for small gaps (up to 5 seconds)
- Automatic data download for larger gaps
- Gap analysis and classification
- Intelligent fill strategy selection
- Data integrity preservation
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .error_handler import handle_errors
from .logger import system_logger
from .pipeline_standards import PipelineStandards, pipeline_standards


class GapType(Enum):
    """Types of data gaps."""

    SMALL = "small"  # <= 5 seconds, use forward fill
    MEDIUM = "medium"  # 5-60 seconds, download data
    LARGE = "large"  # > 60 seconds, download data with warning
    CRITICAL = "critical"  # > 300 seconds, require manual intervention


class GapInfo:
    """Information about a data gap."""

    def __init__(self, start_time: int, end_time: int, gap_size: int, gap_type: GapType):
        self.start_time = start_time
        self.end_time = end_time
        self.gap_size = gap_size
        self.gap_type = gap_type
        self.filled = False
        self.fill_method = None
        self.downloaded_data = None

    def __str__(self):
        return f"Gap({self.start_time} -> {self.end_time}, size={self.gap_size}s, type={self.gap_type.value})"


class EnhancedMissingValueHandler:
    """Enhanced missing value handler with intelligent gap filling."""

    def __init__(self, max_forward_fill_gap: int = 5, download_threshold: int = 5):
        """Initialize enhanced missing value handler."

        Args:
            max_forward_fill_gap: Maximum gap size for forward fill (seconds)
            download_threshold: Gap size threshold for data download (seconds)
        """
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("EnhancedMissingValueHandler")
        self.max_forward_fill_gap = max_forward_fill_gap
        self.download_threshold = download_threshold

        # Gap classification thresholds
        self.gap_thresholds = {
            GapType.SMALL: max_forward_fill_gap,
            GapType.MEDIUM: 60,
            GapType.LARGE: 300,
            GapType.CRITICAL: float("inf"),
        }

        # Fill strategies
        self.fill_strategies = {
            GapType.SMALL: "forward_fill",
            GapType.MEDIUM: "download",
            GapType.LARGE: "download",
            GapType.CRITICAL: "manual_intervention",
        }

    @handle_errors(exceptions=(Exception,), default_return=None, context="missing value handling")
    def handle_missing_values_intelligently(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        symbol: str = None,
        exchange: str = None,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Handle missing values intelligently based on gap size."

        Args:
            data: Data with missing values
            timestamp_column: Name of timestamp column
            symbol: Trading symbol for data download
            exchange: Exchange name for data download
            timeframe: Timeframe for data download

        Returns:
            Data with intelligently filled missing values
        """
        if timestamp_column not in data.columns:
            self.logger.error(f"Timestamp column '{timestamp_column}' not found")
            return data

        # Sort data by timestamp
        data = data.sort_values(timestamp_column).reset_index(drop=True)

        # Analyze gaps
        gaps = self._analyze_gaps(data, timestamp_column)

        if not gaps:
            self.logger.info("No gaps detected in data")
            return data

        # Log gap analysis
        self._log_gap_analysis(gaps)

        # Handle gaps based on type
        filled_data = data.copy()

        for gap in gaps:
            if gap.gap_type == GapType.SMALL:
                filled_data = self._handle_small_gap(filled_data, gap, timestamp_column)
            elif gap.gap_type in [GapType.MEDIUM, GapType.LARGE]:
                if symbol and exchange:
                    filled_data = self._handle_large_gap_with_download(
                        filled_data, gap, timestamp_column, symbol, exchange, timeframe
                    )
                else:
                    self.logger.warning(f"Cannot download data for gap {gap}: missing symbol/exchange")
                    filled_data = self._handle_large_gap_with_fallback(filled_data, gap, timestamp_column)
            elif gap.gap_type == GapType.CRITICAL:
                self.logger.error(f"Critical gap detected: {gap}. Manual intervention required.")
                # For critical gaps, we could raise an exception or use a fallback strategy
                filled_data = self._handle_critical_gap(filled_data, gap, timestamp_column)

        # Final validation
        final_gaps = self._analyze_gaps(filled_data, timestamp_column)
        if final_gaps:
            self.logger.warning(f"Remaining gaps after filling: {len(final_gaps)}")
        else:
            self.logger.info("All gaps successfully filled")

        return filled_data

    def _analyze_gaps(self, data: pd.DataFrame, timestamp_column: str) -> List[GapInfo]:
        """Analyze gaps in the data."

        Args:
            data: Data to analyze
            timestamp_column: Name of timestamp column

        Returns:
            List of gap information
        """
        gaps = []
        timestamps = data[timestamp_column].values

        for i in range(len(timestamps) - 1):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]

            # Calculate expected next time based on timeframe
            # Assuming 1-minute intervals (60 seconds)
            expected_next_time = current_time + 60

            if next_time > expected_next_time:
                gap_size = next_time - expected_next_time
                gap_type = self._classify_gap(gap_size)

                gap = GapInfo(start_time=expected_next_time, end_time=next_time, gap_size=gap_size, gap_type=gap_type)
                gaps.append(gap)

        return gaps

    def _classify_gap(self, gap_size: int) -> GapType:
        """Classify gap based on size."

        Args:
            gap_size: Gap size in seconds

        Returns:
            Gap type
        """
        if gap_size <= self.gap_thresholds[GapType.SMALL]:
            return GapType.SMALL
        elif gap_size <= self.gap_thresholds[GapType.MEDIUM]:
            return GapType.MEDIUM
        elif gap_size <= self.gap_thresholds[GapType.LARGE]:
            return GapType.LARGE
        else:
            return GapType.CRITICAL

    def _log_gap_analysis(self, gaps: List[GapInfo]) -> None:
        """Log gap analysis results."""
        gap_counts = {}
        for gap in gaps:
            gap_type = gap.gap_type.value
            if gap_type not in gap_counts:
                gap_counts[gap_type] = 0
            gap_counts[gap_type] += 1

        self.logger.info(f"Gap analysis: {len(gaps)} total gaps")
        for gap_type, count in gap_counts.items():
            self.logger.info(f"  {gap_type}: {count} gaps")

    def _handle_small_gap(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> pd.DataFrame:
        """Handle small gap with forward fill."

        Args:
            data: Data to fill
            gap: Gap information
            timestamp_column: Name of timestamp column

        Returns:
            Data with small gap filled
        """
        self.logger.info(f"Handling small gap with forward fill: {gap}")

        # Find the row before the gap
        before_gap_idx = data[data[timestamp_column] <= gap.start_time].index[-1]

        # Forward fill the gap
        filled_data = data.copy()

        # Create missing timestamps
        missing_timestamps = []
        current_time = gap.start_time
        while current_time < gap.end_time:
            missing_timestamps.append(current_time)
            current_time += 60  # 1-minute intervals

        # Create new rows with forward-filled values
        new_rows = []
        for timestamp in missing_timestamps:
            new_row = data.iloc[before_gap_idx].copy()
            new_row[timestamp_column] = timestamp
            new_rows.append(new_row)

        # Insert new rows
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            filled_data = pd.concat([filled_data, new_df], ignore_index=True)
            filled_data = filled_data.sort_values(timestamp_column).reset_index(drop=True)

        gap.filled = True
        gap.fill_method = "forward_fill"

        return filled_data

    def _handle_large_gap_with_download(
        self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str, symbol: str, exchange: str, timeframe: str
    ) -> pd.DataFrame:
        """Handle large gap by downloading missing data."

        Args:
            data: Data to fill
            gap: Gap information
            timestamp_column: Name of timestamp column
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Data with downloaded data filling the gap
        """
        self.logger.info(f"Downloading data for gap: {gap}")

        try:
            # Download missing data
            downloaded_data = self._download_missing_data(symbol, exchange, timeframe, gap.start_time, gap.end_time)

            if downloaded_data is not None and len(downloaded_data) > 0:
                # Insert downloaded data
                filled_data = self._insert_downloaded_data(data, downloaded_data, timestamp_column)

                gap.filled = True
                gap.fill_method = "download"
                gap.downloaded_data = downloaded_data

                self.logger.info(f"Successfully downloaded and inserted {len(downloaded_data)} rows")
                return filled_data
            else:
                self.logger.warning(f"No data downloaded for gap {gap}, using fallback")
                return self._handle_large_gap_with_fallback(data, gap, timestamp_column)

        except Exception as e:
            self.logger.error(f"Failed to download data for gap {gap}: {e}")
            return self._handle_large_gap_with_fallback(data, gap, timestamp_column)

    def _download_missing_data(
        self, symbol: str, exchange: str, timeframe: str, start_time: int, end_time: int
    ) -> Optional[pd.DataFrame]:
        """Download missing data from exchange."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Downloaded data or None if failed
        """
        try:
            # Convert timestamps to datetime
            start_dt = datetime.fromtimestamp(start_time)
            end_dt = datetime.fromtimestamp(end_time)

            self.logger.info(f"Downloading {symbol} data from {exchange} for {start_dt} to {end_dt}")

            # Import exchange-specific downloader
            if exchange.lower() == "binance":
                from src.training.steps.data_downloader import DataDownloader
        except Exception as e:
            pass  # TODO: Handle exception properly
import copy

downloader = DataDownloader()

                # Download klines data
                downloaded_data = downloader.download_klines(
                    symbol=symbol, interval=timeframe, start_time=start_dt, end_time=end_dt
                )

                if downloaded_data is not None and len(downloaded_data) > 0:
                    # Ensure timestamp column is int64
                    downloaded_data["timestamp"] = (
                        pd.to_datetime(downloaded_data["timestamp"]).astype(np.int64) // 10**9
                    )

                    return downloaded_data
                else:
                    self.logger.warning("No data returned from downloader")
                    return None
            else:
                self.logger.warning(f"Exchange {exchange} not supported for data download")
                return None

        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            return None

    def _insert_downloaded_data(
        self, data: pd.DataFrame, downloaded_data: pd.DataFrame, timestamp_column: str
    ) -> pd.DataFrame:
        """Insert downloaded data into the main dataset."

        Args:
            data: Main dataset
            downloaded_data: Downloaded data to insert
            timestamp_column: Name of timestamp column

        Returns:
            Data with downloaded data inserted
        """
        # Combine datasets
        combined_data = pd.concat([data, downloaded_data], ignore_index=True)

        # Sort by timestamp and remove duplicates
        combined_data = combined_data.sort_values(timestamp_column).reset_index(drop=True)
        combined_data = combined_data.drop_duplicates(subset=[timestamp_column])

        return combined_data

    def _handle_large_gap_with_fallback(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> pd.DataFrame:
        """Handle large gap with fallback strategy (interpolation)."

        Args:
            data: Data to fill
            gap: Gap information
            timestamp_column: Name of timestamp column

        Returns:
            Data with gap filled using fallback strategy
        """
        self.logger.info(f"Using fallback strategy for gap: {gap}")

        # Use interpolation as fallback
        filled_data = data.copy()

        # Find the rows before and after the gap
        before_gap_idx = data[data[timestamp_column] <= gap.start_time].index[-1]
        after_gap_idx = data[data[timestamp_column] >= gap.end_time].index[0]

        # Create missing timestamps
        missing_timestamps = []
        current_time = gap.start_time
        while current_time < gap.end_time:
            missing_timestamps.append(current_time)
            current_time += 60  # 1-minute intervals

        # Interpolate values for each column
        for timestamp in missing_timestamps:
            # Calculate interpolation weight
            time_diff = timestamp - data.iloc[before_gap_idx][timestamp_column]
            total_gap = data.iloc[after_gap_idx][timestamp_column] - data.iloc[before_gap_idx][timestamp_column]
            weight = time_diff / total_gap if total_gap > 0 else 0

            # Create interpolated row
            new_row = data.iloc[before_gap_idx].copy()
            new_row[timestamp_column] = timestamp

            # Interpolate numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != timestamp_column:
                    before_val = data.iloc[before_gap_idx][col]
                    after_val = data.iloc[after_gap_idx][col]
                    interpolated_val = before_val + weight * (after_val - before_val)
                    new_row[col] = interpolated_val

            # Add to dataset
            filled_data = pd.concat([filled_data, pd.DataFrame([new_row])], ignore_index=True)

        # Sort and reset index
        filled_data = filled_data.sort_values(timestamp_column).reset_index(drop=True)

        gap.filled = True
        gap.fill_method = "interpolation_fallback"

        return filled_data

    def _handle_critical_gap(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> pd.DataFrame:
        """Handle critical gap (requires manual intervention)."

        Args:
            data: Data to fill
            gap: Gap information
            timestamp_column: Name of timestamp column

        Returns:
            Data with critical gap handled
        """
        self.logger.error(f"Critical gap detected: {gap}")
        self.logger.error("Manual intervention required for critical gaps")

        # For now, use the same fallback as large gaps
        # In a production system, this might raise an exception or trigger alerts
        return self._handle_large_gap_with_fallback(data, gap, timestamp_column)

    def get_gap_report(self, data: pd.DataFrame, timestamp_column: str = "timestamp") -> Dict[str, Any]:
        """Generate gap analysis report."

        Args:
            data: Data to analyze
            timestamp_column: Name of timestamp column

        Returns:
            Gap analysis report
        """
        gaps = self._analyze_gaps(data, timestamp_column)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_gaps": len(gaps),
            "gap_summary": {},
            "gap_details": [],
        }

        # Summarize gaps by type
        for gap_type in GapType:
            gap_type_gaps = [g for g in gaps if g.gap_type == gap_type]
            report["gap_summary"][gap_type.value] = {
                "count": len(gap_type_gaps),
                "total_size": sum(g.gap_size for g in gap_type_gaps),
                "avg_size": np.mean([g.gap_size for g in gap_type_gaps]) if gap_type_gaps else 0,
            }

        # Detailed gap information
        for gap in gaps:
            report["gap_details"].append(
                {
                    "start_time": gap.start_time,
                    "end_time": gap.end_time,
                    "gap_size": gap.gap_size,
                    "gap_type": gap.gap_type.value,
                    "filled": gap.filled,
                    "fill_method": gap.fill_method,
                }
            )

        return report

    def validate_data_continuity(
        self, data: pd.DataFrame, timestamp_column: str = "timestamp", expected_interval: int = 60
    ) -> Dict[str, Any]:
        """Validate data continuity and identify issues."

        Args:
            data: Data to validate
            timestamp_column: Name of timestamp column
            expected_interval: Expected interval between timestamps (seconds)

        Returns:
            Continuity validation report
        """
        if timestamp_column not in data.columns:
            return {"valid": False, "error": f"Timestamp column '{timestamp_column}' not found"}

        # Sort data by timestamp
        data = data.sort_values(timestamp_column).reset_index(drop=True)
        timestamps = data[timestamp_column].values

        issues = []
        total_intervals = len(timestamps) - 1

        for i in range(total_intervals):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            interval = next_time - current_time

            if interval != expected_interval:
                issues.append(
                    {
                        "position": i,
                        "current_time": current_time,
                        "next_time": next_time,
                        "actual_interval": interval,
                        "expected_interval": expected_interval,
                        "deviation": interval - expected_interval,
                    }
                )

        report = {
            "timestamp": datetime.now().isoformat(),
            "valid": len(issues) == 0,
            "total_intervals": total_intervals,
            "issues_count": len(issues),
            "issues": issues,
            "continuity_score": 1 - (len(issues) / total_intervals) if total_intervals > 0 else 1.0,
        }

        return report


# Global enhanced missing value handler instance
enhanced_missing_value_handler = EnhancedMissingValueHandler()
