#!/usr/bin/env python3
"""
Unified Gap Filler

This module provides centralized gap filling functionality that integrates with:
- Unified data downloader for re-downloading missing data
- Data quality detection for gap identification
- Standardized file paths for compatibility with the rest of the codebase

Consolidates gap filling logic from multiple files into a single, comprehensive implementation.
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
            step_name="gap_filling",
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

# Import the unified downloader and gap collection hook
from .unified_data_downloader import UnifiedDataDownloader

logger = system_logger.getChild("UnifiedGapFiller")

class UnifiedGapFiller:
    """Unified gap filler that integrates with the unified downloader and maintains compatibility with existing file paths."""

    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('UnifiedGapFiller')

        # Initialize unified downloader for gap filling
        self.downloader = UnifiedDataDownloader(data_cache_path)

        # Initialize gap collection hook for automatic data re-collection
        try:
            from src.utils.data.quality.gap_collection_hook import get_gap_collection_hook
            self.gap_collection_hook = get_gap_collection_hook()
            self.logger.info("✅ Gap collection hook initialized")
        except ImportError as e:
            self.logger.warning(f"⚠️ Gap collection hook not available: {e}")
            self.gap_collection_hook = None

        # Gap detection thresholds (in seconds) - updated to match new requirements
        self.gap_thresholds = {
            'aggtrades': 0.5,   # 0.5 seconds for aggtrades - triggers re-download
            'klines': 66,       # 1.1 minutes for 1m klines - triggers re-download (avoid unnecessary downloads for 1m data)
            'futures': 32400    # 9 hours for futures - triggers re-download
        }

        # Gap filling statistics
        self.gap_stats = {
            'total_gaps_detected': 0,
            'total_gaps_filled': 0,
            'total_gaps_failed': 0,
            'total_rows_downloaded': 0,
            'start_time': None
        }

    @handles_errors(context="detect_gaps")
    @log_all_calls
    def detect_gaps(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in data files for a specific symbol, exchange, and data type.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            start_date: Start date for gap detection
            end_date: End date for gap detection

        Returns:
            List of gap information dictionaries
        """
        self.logger.info(f"🔍 Detecting gaps in {exchange}_{symbol}_{data_type}")

        if data_type not in self.gap_thresholds:
            self.logger.error(f"❌ Unsupported data type: {data_type}")
            return []

        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()

            # Get file paths based on data type
            file_paths = self._get_data_file_paths(symbol, exchange, data_type, start_date, end_date)

            if not file_paths:
                self.logger.warning(f"⚠️ No data files found for {exchange}_{symbol}_{data_type}")
                return []

            all_gaps = []
            threshold = self.gap_thresholds[data_type]

            # Detect gaps in each file
            for file_path in file_paths:
                gaps = self._detect_gaps_in_file(file_path, data_type, threshold)
                all_gaps.extend(gaps)

            # Update statistics
            self.gap_stats['total_gaps_detected'] += len(all_gaps)

            self.logger.info(f"✅ Detected {len(all_gaps)} gaps in {exchange}_{symbol}_{data_type}")
            return all_gaps

        except Exception as e:
            self.logger.exception(f"❌ Error detecting gaps: {e}")
            return []

    @handles_errors(context="fill_gaps")
    @log_all_calls
    async def fill_gaps(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        gaps: List[Dict[str, Any]],
        max_concurrent_downloads: int = 3
    ) -> Dict[str, Any]:
        """
        Fill detected gaps by re-downloading missing data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            gaps: List of gap information from detect_gaps()
            max_concurrent_downloads: Maximum concurrent downloads

        Returns:
            Dictionary with gap filling results
        """
        self.logger.info(f"🔧 Filling {len(gaps)} gaps for {exchange}_{symbol}_{data_type}")

        if not gaps:
            return {
                'success': True,
                'gaps_filled': 0,
                'gaps_failed': 0,
                'rows_downloaded': 0,
                'errors': []
            }

        results = {
            'success': True,
            'gaps_filled': 0,
            'gaps_failed': 0,
            'rows_downloaded': 0,
            'errors': []
        }

        try:
            # Process gaps in batches to avoid overwhelming the API
            semaphore = asyncio.Semaphore(max_concurrent_downloads)

            async def fill_single_gap(gap_info: Dict[str, Any]) -> Tuple[bool, int, Optional[str]]:
                async with semaphore:
                    return await self._fill_single_gap(symbol, exchange, data_type, gap_info)

            # Process all gaps concurrently
            tasks = [fill_single_gap(gap) for gap in gaps]
            gap_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(gap_results):
                if isinstance(result, Exception):
                    results['gaps_failed'] += 1
                    results['errors'].append(f"Gap {i}: {str(result)}")
                    self.logger.error(f"❌ Error filling gap {i}: {result}")
                else:
                    success, rows_downloaded, error = result
                    if success:
                        results['gaps_filled'] += 1
                        results['rows_downloaded'] += rows_downloaded
                    else:
                        results['gaps_failed'] += 1
                        if error:
                            results['errors'].append(f"Gap {i}: {error}")

            # Update statistics
            self.gap_stats['total_gaps_filled'] += results['gaps_filled']
            self.gap_stats['total_gaps_failed'] += results['gaps_failed']
            self.gap_stats['total_rows_downloaded'] += results['rows_downloaded']

            self.logger.info(f"✅ Gap filling completed: {results['gaps_filled']} filled, {results['gaps_failed']} failed")
            return results

        except Exception as e:
            self.logger.exception(f"❌ Error filling gaps: {e}")
            return {
                'success': False,
                'gaps_filled': 0,
                'gaps_failed': len(gaps),
                'rows_downloaded': 0,
                'errors': [str(e)]
            }

    @handles_errors(context="detect_and_fill_gaps")
    @log_all_calls
    async def detect_and_fill_gaps(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        auto_fill: bool = True
    ) -> Dict[str, Any]:
        """
        Detect and optionally fill gaps in data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            start_date: Start date for gap detection
            end_date: End date for gap detection
            auto_fill: Whether to automatically fill detected gaps

        Returns:
            Dictionary with detection and filling results
        """
        self.logger.info(f"🔍 Detecting and filling gaps for {exchange}_{symbol}_{data_type}")

        try:
            # Detect gaps
            gaps = self.detect_gaps(symbol, exchange, data_type, start_date, end_date)

            if not gaps:
                return {
                    'success': True,
                    'gaps_detected': 0,
                    'gaps_filled': 0,
                    'gaps_failed': 0,
                    'rows_downloaded': 0,
                    'message': 'No gaps detected'
                }

            # Fill gaps if requested
            fill_results = {}
            if auto_fill:
                # Check if gaps exceed thresholds and trigger collection hook
                large_gaps = [gap for gap in gaps if gap.get('gap_size', 0) >= self.gap_thresholds.get(data_type, 300)]

                if large_gaps and self.gap_collection_hook:
                    self.logger.info(f"🔄 Large gaps detected ({len(large_gaps)}), triggering collection hook...")
                    for gap in large_gaps:
                        try:
                            collection_result = self.gap_collection_hook.trigger_data_collection(
                                gap, data_type, symbol, exchange
                            )
                            if collection_result.get('triggered', False):
                                self.logger.info(f"✅ Collection hook triggered for gap: {gap}")
                            else:
                                self.logger.warning(f"⚠️ Collection hook not triggered: {collection_result.get('reason', 'Unknown')}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to trigger collection hook: {e}")

                fill_results = await self.fill_gaps(symbol, exchange, data_type, gaps)
            else:
                fill_results = {
                    'gaps_filled': 0,
                    'gaps_failed': 0,
                    'rows_downloaded': 0,
                    'errors': []
                }

            return {
                'success': True,
                'gaps_detected': len(gaps),
                'gaps_filled': fill_results['gaps_filled'],
                'gaps_failed': fill_results['gaps_failed'],
                'rows_downloaded': fill_results['rows_downloaded'],
                'errors': fill_results.get('errors', [])
            }

        except Exception as e:
            self.logger.exception(f"❌ Error in detect_and_fill_gaps: {e}")
            return {
                'success': False,
                'gaps_detected': 0,
                'gaps_filled': 0,
                'gaps_failed': 0,
                'rows_downloaded': 0,
                'errors': [str(e)]
            }

    def _get_data_file_paths(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Path]:
        """Get file paths for data files based on data type and date range."""
        file_paths = []

        try:
            if data_type == 'aggtrades':
                # Aggtrades files: aggtrades_BINANCE_ETHUSDT_20250101.parquet
                pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"
                file_paths = list(self.data_cache_path.glob(pattern))

            elif data_type == 'klines':
                # Klines files: klines_BINANCE_ETHUSDT_1m_202501.parquet
                pattern = f"klines_{exchange}_{symbol}_1m_*.parquet"
                file_paths = list(self.data_cache_path.glob(pattern))

            elif data_type == 'futures':
                # Futures files: futures_BINANCE_ETHUSDT_202501.parquet
                pattern = f"futures_{exchange}_{symbol}_*.parquet"
                file_paths = list(self.data_cache_path.glob(pattern))

            # Filter by date range
            filtered_paths = []
            for file_path in file_paths:
                if self._file_in_date_range(file_path, start_date, end_date):
                    filtered_paths.append(file_path)

            return filtered_paths

        except Exception as e:
            self.logger.exception(f"❌ Error getting file paths: {e}")
            return []

    def _file_in_date_range(self, file_path: Path, start_date: datetime, end_date: datetime) -> bool:
        """Check if file is within the specified date range."""
        try:
            # Extract date from filename
            filename = file_path.stem

            if 'aggtrades' in filename:
                # aggtrades_BINANCE_ETHUSDT_20250101
                date_str = filename.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m%d')

            elif 'klines' in filename:
                # klines_BINANCE_ETHUSDT_1m_202501
                date_str = filename.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m')

            elif 'futures' in filename:
                # futures_BINANCE_ETHUSDT_202501
                date_str = filename.split('_')[-1]
                file_date = datetime.strptime(date_str, '%Y%m')

            else:
                return True  # Include unknown formats

            return start_date <= file_date <= end_date

        except Exception:
            return True  # Include files with unparseable dates

    def _detect_gaps_in_file(
        self,
        file_path: Path,
        data_type: str,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect gaps in a single data file."""
        try:
            # Read file using utils/ safe operations
            df = safe_read_parquet(file_path)
            if df is None or df.empty:
                return []

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                return []

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            # Calculate time differences
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

            # Find gaps based on data type
            if data_type == 'aggtrades':
                gap_rows = df[df['time_diff'] > threshold]
            elif data_type == 'klines':
                df['time_diff'] = df['time_diff'] / 60  # Convert to minutes
                gap_rows = df[df['time_diff'] > threshold]
            elif data_type == 'futures':
                df['time_diff'] = df['time_diff'] / 3600  # Convert to hours
                gap_rows = df[df['time_diff'] > threshold]
            else:
                gap_rows = df[df['time_diff'] > threshold]

            # Extract gap information
            gaps = []
            for idx, row in gap_rows.iterrows():
                if idx > 0:
                    gap_start = df.loc[idx - 1, 'timestamp']
                    gap_end = row['timestamp']
                    gap_duration = (gap_end - gap_start).total_seconds()

                    gaps.append({
                        'file': file_path.name,
                        'gap_start': gap_start,
                        'gap_end': gap_end,
                        'gap_duration_seconds': gap_duration,
                        'data_type': data_type,
                        'threshold': threshold
                    })

            return gaps

        except Exception as e:
            self.logger.exception(f"❌ Error detecting gaps in {file_path}: {e}")
            return []

    async def _fill_single_gap(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        gap_info: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:
        """Fill a single gap by downloading missing data."""
        try:
            gap_start = gap_info['gap_start']
            gap_end = gap_info['gap_end']

            self.logger.info(f"🔧 Filling gap: {gap_start} to {gap_end}")

            # Download missing data using unified downloader
            if data_type == 'klines':
                success, data, error = await self.downloader.download_klines(
                    symbol, exchange, "1m", gap_start, gap_end
                )
            elif data_type == 'aggtrades':
                # Skip aggtrades downloads as per new setup - only klines are needed
                self.logger.info(f"⚠️ Skipping aggtrades download for {symbol} - aggtrades downloads disabled")
                return False, 0, "Aggtrades downloads disabled - only klines are processed"
            else:
                return False, 0, f"Unsupported data type: {data_type}"

            if not success:
                return False, 0, error

            # Save downloaded data
            if data:
                save_success = await self._save_gap_data(data, symbol, exchange, data_type, gap_start)
                if save_success:
                    return True, len(data), None
                else:
                    return False, 0, "Failed to save downloaded data"
            else:
                return False, 0, "No data downloaded"

        except Exception as e:
            self.logger.exception(f"❌ Error filling gap: {e}")
            return False, 0, str(e)

    async def _save_gap_data(
        self,
        data: List[Dict[str, Any]],
        symbol: str,
        exchange: str,
        data_type: str,
        gap_start: datetime
    ) -> bool:
        """Save downloaded gap data to appropriate file."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Generate filename based on data type and date
            if data_type == 'aggtrades':
                date_str = gap_start.strftime('%Y%m%d')
                filename = f"aggtrades_{exchange}_{symbol}_{date_str}.parquet"
            elif data_type == 'klines':
                date_str = gap_start.strftime('%Y%m')
                filename = f"klines_{exchange}_{symbol}_1m_{date_str}.parquet"
            elif data_type == 'futures':
                date_str = gap_start.strftime('%Y%m')
                filename = f"futures_{exchange}_{symbol}_{date_str}.parquet"
            else:
                return False

            file_path = self.data_cache_path / filename

            # Use utils/ safe operations
            success = safe_to_parquet(df, file_path)

            if success:
                self.logger.info(f"💾 Saved gap data: {filename}")

            return success

        except Exception as e:
            self.logger.exception(f"❌ Error saving gap data: {e}")
            return False

    def get_gap_stats(self) -> Dict[str, Any]:
        """Get gap filling statistics."""
        return {
            **self.gap_stats,
            'success_rate': (
                self.gap_stats['total_gaps_filled'] /
                max(self.gap_stats['total_gaps_detected'], 1) * 100
            )
        }

    def reset_stats(self):
        """Reset gap filling statistics."""
        self.gap_stats = {
            'total_gaps_detected': 0,
            'total_gaps_filled': 0,
            'total_gaps_failed': 0,
            'total_rows_downloaded': 0,
            'start_time': None
        }

# Convenience functions for backward compatibility
@handles_errors()
def detect_gaps(symbol: str, exchange: str, data_type: str, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function for detecting gaps."""
    gap_filler = UnifiedGapFiller()
    return gap_filler.detect_gaps(symbol, exchange, data_type, **kwargs)

@handles_errors()
async def fill_gaps(symbol: str, exchange: str, data_type: str, gaps: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Convenience function for filling gaps."""
    gap_filler = UnifiedGapFiller()
    return await gap_filler.fill_gaps(symbol, exchange, data_type, gaps, **kwargs)

@handles_errors()
async def detect_and_fill_gaps(symbol: str, exchange: str, data_type: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for detecting and filling gaps."""
    gap_filler = UnifiedGapFiller()
    return await gap_filler.detect_and_fill_gaps(symbol, exchange, data_type, **kwargs)
