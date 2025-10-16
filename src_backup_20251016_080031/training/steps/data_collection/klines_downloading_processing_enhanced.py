"""
Enhanced Klines Data Downloading and Processing Pipeline

This module provides a complete pipeline for downloading, processing, and quality-checking
historical klines data with gap detection, duplicate handling, and column management.

Features:
- Download historical klines data using HistoricalDataPipeline
- Detect and fill gaps > 1m
- Identify and handle duplicate timestamps (warn on false duplicates, remove true duplicates)
- Remove unwanted columns (taker_buy_base, taker_buy_quote, year)
- Comprehensive data quality checks
- Exchange-agnostic data standardization
- Fast fail pattern with no fallbacks or mock data
- Full type hints and tprint logging
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import sys
import asyncio
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.data.quality.comprehensive_duplicate_analyzer import (
    ComprehensiveDuplicateAnalyzer,
    analyze_duplicates_comprehensive
)
from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface
from src.utils.data.gap_detector import GapDetector
from exchanges.shared.unified_ohlcv_standardizer import UnifiedExchangeStandardizer, ExchangeType


class KlinesDataProcessingPipeline:
    """Complete pipeline for downloading, processing, and quality-checking klines data."""

    def __init__(self, data_dir: str = "historical_data") -> None:
        """Initialize the processing pipeline.

        Args:
            data_dir: Base directory for historical data
        """
        self.data_dir = data_dir
        # Removed logger - using tprint instead

        # Initialize components
        self.historical_pipeline = HistoricalDataPipeline(data_dir)
        self.gap_detector = GapDetector(data_dir)
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer()
        self.data_standardizer = UnifiedExchangeStandardizer()

        # Quality checker will be initialized when first used
        self._quality_checker: Optional[KlinesDataQualityChecker] = None

        # Columns to remove
        self.columns_to_remove: List[str] = ['taker_buy_base', 'taker_buy_quote', 'year']

    @property
    def quality_checker(self) -> 'KlinesDataQualityChecker':
        """Lazy initialization of quality checker."""
        if self._quality_checker is None:
            self._quality_checker = KlinesDataQualityChecker(self.data_dir)
        return self._quality_checker

    def standardize_data_format(self, df: pd.DataFrame, symbol: str, interval: str, exchange: str = "binance") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Standardize data format using centralized standardizer.
        
        Args:
            df: Raw DataFrame from exchange
            symbol: Trading symbol
            interval: Data interval
            exchange: Exchange name
            
        Returns:
            Tuple of (standardized_dataframe, standardization_report)
        """
        try:
            standardized_df = self.data_standardizer.standardize_to_dataframe(
                df, ExchangeType(exchange.upper()), symbol, interval
            )
            
            tprint_success(f"✅ Data standardized: {len(standardized_df)} records for {symbol} {interval}")
            
            return standardized_df, {"success": True, "shape": standardized_df.shape}
            
        except Exception as e:
            tprint_error(f"❌ Failed to standardize data format: {e}")
            raise  # Fast fail - no fallback

    def create_consolidated_features_file(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        exchange: str = "binance"
    ) -> Dict[str, Any]:
        """Create a consolidated features parquet file with required columns.

        This creates a file with the format: historical_data/features_binance_{SYMBOL}_consolidated.parquet
        containing the required columns: ['timestamp', 'exchange', 'timeframe']

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Time interval (e.g., "1m")
            exchange: Exchange name (default: "binance")

        Returns:
            Dictionary with consolidation results
        """
        try:
            tprint_info(f"📦 Creating consolidated features file for {symbol} {interval}")

            # Define output file path
            output_file = Path(self.data_dir) / f"features_{exchange.lower()}_{symbol.upper()}_consolidated.parquet"

            # Find processed data files
            data_path = Path(self.data_dir) / "binance" / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                error_msg = f"Processed data directory not found: {data_path}"
                tprint_error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "files_processed": 0,
                    "total_records": 0
                }

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                error_msg = "No parquet files found in processed directory"
                tprint_error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "files_processed": 0,
                    "total_records": 0
                }

            tprint_info(f"🔍 Found {len(parquet_files)} parquet files to consolidate")

            # Combine all data
            combined_data: List[pd.DataFrame] = []
            total_records = 0

            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)

                    # Add required columns
                    df = df.copy()
                    df['exchange'] = exchange
                    df['timeframe'] = interval

                    # Ensure timestamp column exists and is in correct format
                    if 'timestamp' not in df.columns:
                        if hasattr(df.index, 'name') and df.index.name == 'timestamp':
                            df = df.reset_index()
                        elif df.index.name is None and df.index.dtype in ['int64', 'datetime64[ns]']:
                            # Index appears to be timestamp but not named
                            df.index.name = 'timestamp'
                            df = df.reset_index()
                        else:
                            # Try to infer timestamp from index
                            df['timestamp'] = df.index

                    combined_data.append(df)
                    total_records += len(df)

                    tprint(f"✅ Processed {file_path.name}: {len(df)} records", "DEBUG")

                except Exception as e:
                    tprint_error(f"❌ Error processing {file_path.name}: {e}")
                    # Fast fail - don't continue with corrupted files
                    raise

            if not combined_data:
                error_msg = "Failed to process any data files"
                tprint_error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "files_processed": 0,
                    "total_records": 0
                }

            # Concatenate all data
            consolidated_df = pd.concat(combined_data, ignore_index=True)

            # Remove any duplicate records based on timestamp
            initial_records = len(consolidated_df)
            consolidated_df = consolidated_df.drop_duplicates(subset=['timestamp'], keep='first')
            duplicates_removed = initial_records - len(consolidated_df)

            # Ensure required columns are present and in correct order
            required_cols = ['timestamp', 'exchange', 'timeframe']
            existing_cols = [col for col in required_cols if col in consolidated_df.columns]
            other_cols = [col for col in consolidated_df.columns if col not in required_cols]

            # Reorder columns with required columns first
            final_column_order = existing_cols + other_cols
            consolidated_df = consolidated_df[final_column_order]

            # Sort by timestamp
            consolidated_df = consolidated_df.sort_values('timestamp').reset_index(drop=True)

            # Save consolidated file
            consolidated_df.to_parquet(output_file, index=False, compression='snappy')

            # Create results summary
            result = {
                "success": True,
                "output_file": str(output_file),
                "files_processed": len(combined_data),
                "total_records": len(consolidated_df),
                "duplicates_removed": duplicates_removed,
                "columns": list(consolidated_df.columns),
                "required_columns_present": all(col in consolidated_df.columns for col in required_cols),
                "file_size_mb": round(output_file.stat().st_size / (1024 * 1024), 2),
                "date_range": {
                    "start": consolidated_df['timestamp'].min(),
                    "end": consolidated_df['timestamp'].max()
                }
            }

            tprint_success("✅ Consolidated features file created successfully")
            tprint_info(f"   📁 Output: {output_file}")
            tprint_info(f"   📊 Records: {len(consolidated_df):,}")
            tprint_info(f"   🗂️  Columns: {len(consolidated_df.columns)}")
            tprint_info(f"   📏 Size: {result['file_size_mb']} MB")

            return result

        except Exception as e:
            tprint_error(f"❌ Failed to create consolidated features file: {e}")
            raise  # Fast fail - no fallback

    async def run_complete_pipeline(
        self,
        symbol: str = "ETHUSDT",
        years: Optional[int] = None,
        interval: str = "1m",
        api_key: str = "",
        api_secret: str = "",
        max_gap_minutes: int = 1,
        create_consolidated: bool = True
    ) -> Dict[str, Any]:
        """Run the complete klines processing pipeline.

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT", default: "ETHUSDT")
            years: Number of years of data to download (default: from centralized config)
            interval: Kline interval (e.g., "1m")
            api_key: Binance API key
            api_secret: Binance API secret
            max_gap_minutes: Maximum allowed gap in minutes
            create_consolidated: Whether to create consolidated features file

        Returns:
            Dictionary with pipeline results
        """
        try:
            # Use centralized configuration if years not specified
            if years is None:
                from src.config.pipeline_modes import get_full_mode_config
                mode_config = get_full_mode_config()
                years = mode_config.lookback_years
            
            tprint_info(f"🚀 Starting complete klines processing pipeline for {symbol}")

            results = {
                "symbol": symbol,
                "years": years,
                "interval": interval,
                "steps_completed": [],
                "errors": [],
                "warnings": [],
                "summary": {}
            }

            # Step 1: Download data using HistoricalDataPipeline
            tprint_info(f"📥 Step 1: Downloading {years} years of {symbol} {interval} data")
            
            # Create ExchangeInterface for exchange-agnostic data access
            exchange_interface: Optional[ExchangeInterface] = None
            if api_key and api_secret:
                exchange_config = {
                    'exchange_type': 'binance',
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'testnet': False
                }
                exchange_interface = create_exchange_interface(exchange_config)
                await exchange_interface.connect()
            else:
                # Fast fail - require API credentials
                error_msg = "API credentials are required for data download"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results
            
            download_results = await self.historical_pipeline.run_complete_pipeline(
                symbol=symbol,
                years=years,
                exchange_interface=exchange_interface,
                api_key=api_key,
                api_secret=api_secret,
                target_intervals=[interval] if interval != "1m" else []
            )

            if not download_results.get("pipeline_success", False):
                error_msg = f"Data download failed: {download_results.get('errors', [])}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("download")
            results["summary"]["download"] = download_results
            tprint_success("✅ Data download completed")

            # Step 2: Remove unwanted columns
            tprint_info("🧹 Step 2: Removing unwanted columns")
            column_removal_results = self.remove_unwanted_columns(symbol, interval)
            if not column_removal_results.get("success", False):
                error_msg = f"Column removal failed: {column_removal_results.get('error', 'Unknown error')}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("column_removal")
            results["summary"]["column_removal"] = column_removal_results

            # Step 2.5: Add required columns to processed files
            tprint_info("📦 Step 2.5: Adding required columns (exchange, timeframe) to processed files")
            column_addition_results = self.add_required_columns_to_processed_files(symbol, interval, "binance")
            if not column_addition_results.get("success", False):
                error_msg = f"Column addition failed: {column_addition_results.get('error', 'Unknown error')}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("column_addition")
            results["summary"]["column_addition"] = column_addition_results

            # Step 3: Detect and fill gaps > 1m
            tprint_info("🔍 Step 3: Detecting gaps > 1m and re-downloading if needed")
            gap_results = await self.handle_gaps_with_column_removal(
                symbol, interval, max_gap_minutes, exchange_interface, api_key, api_secret
            )
            if gap_results.get("error"):
                error_msg = f"Gap handling failed: {gap_results.get('error')}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("gap_handling")
            results["summary"]["gap_handling"] = gap_results

            # Step 4: Handle duplicates (warn on false, remove true)
            tprint_info("🔍 Step 4: Analyzing and handling duplicate timestamps")
            duplicate_results = self.handle_duplicates(symbol, interval)
            if not duplicate_results.get("success", False):
                error_msg = f"Duplicate handling failed: {duplicate_results.get('error', 'Unknown error')}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("duplicate_handling")
            results["summary"]["duplicate_handling"] = duplicate_results

            # Add warnings from duplicate analysis
            if duplicate_results.get("warnings"):
                results["warnings"].extend(duplicate_results["warnings"])

            # Step 5: Final quality check
            tprint_info("✅ Step 5: Running final data quality check")
            quality_results = self.quality_checker.check_processed_data_quality(
                symbol, [interval]
            )
            if not quality_results.get("overall_quality", False):
                error_msg = f"Data quality check failed: {quality_results.get('issues', [])}"
                tprint_error(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                return results

            results["steps_completed"].append("quality_check")
            results["summary"]["quality_check"] = quality_results

            # Step 6: Create consolidated features file (optional)
            if create_consolidated:
                tprint_info("📦 Step 6: Creating consolidated features file")
                try:
                    consolidated_results = self.create_consolidated_features_file(
                        symbol=symbol,
                        interval=interval,
                        exchange="binance"
                    )
                    results["steps_completed"].append("consolidated_file_creation")
                    results["summary"]["consolidated_file_creation"] = consolidated_results
                    tprint_success("✅ Consolidated features file created successfully")
                except Exception as e:
                    error_msg = f"Consolidated file creation failed: {str(e)}"
                    tprint_error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)
                    return results

            # Cleanup exchange interface
            if exchange_interface:
                try:
                    await exchange_interface.disconnect()
                except Exception as e:
                    tprint_warning(f"Error disconnecting exchange interface: {e}")

            # Overall success
            results["pipeline_success"] = len(results["errors"]) == 0
            results["completion_time"] = datetime.now().isoformat()

            tprint_success(f"🎉 Pipeline completed: {len(results['steps_completed'])} steps, {len(results['errors'])} errors, {len(results['warnings'])} warnings")

            return results

        except Exception as e:
            tprint_error(f"❌ Pipeline failed: {e}")
            results["errors"].append(str(e))
            results["pipeline_success"] = False
            return results

    def remove_unwanted_columns(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Remove unwanted columns from all data files.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Dictionary with column removal results
        """
        try:
            tprint_info(f"🧹 Removing columns {self.columns_to_remove} from {symbol} {interval} data")

            # Get data directory
            data_path = Path(self.data_dir) / "binance" / symbol.lower() / "raw" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                error_msg = "No data directory found"
                tprint_error(f"❌ {error_msg}")
                return {"files_processed": 0, "columns_removed": 0, "success": False, "error": error_msg}

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                error_msg = "No parquet files found"
                tprint_error(f"❌ {error_msg}")
                return {"files_processed": 0, "columns_removed": 0, "success": False, "error": error_msg}

            files_processed = 0
            columns_removed_total = 0

            for file_path in parquet_files:
                try:
                    # Read the file
                    df = pd.read_parquet(file_path)

                    # Track columns before removal
                    columns_before = set(df.columns)
                    columns_to_remove_present = [col for col in self.columns_to_remove if col in df.columns]

                    if columns_to_remove_present:
                        # Remove the unwanted columns
                        df = df.drop(columns=columns_to_remove_present)

                        # Save the modified file
                        df.to_parquet(file_path, index=True, compression='snappy')

                        columns_removed = len(columns_to_remove_present)
                        columns_removed_total += columns_removed

                        tprint(f"✅ Removed {columns_removed} columns from {file_path.name}", "DEBUG")
                    else:
                        tprint(f"ℹ️ No unwanted columns found in {file_path.name}", "DEBUG")

                    files_processed += 1

                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {e}"
                    tprint_error(f"❌ {error_msg}")
                    return {"files_processed": 0, "columns_removed": 0, "success": False, "error": error_msg}

            result = {
                "files_processed": files_processed,
                "columns_removed": columns_removed_total,
                "columns_targeted": self.columns_to_remove,
                "success": True,
                "message": f"Processed {files_processed} files, removed {columns_removed_total} column instances"
            }

            tprint_success(f"✅ Column removal completed: {result['message']}")
            return result

        except Exception as e:
            error_msg = f"Column removal failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {
                "files_processed": 0,
                "columns_removed": 0,
                "success": False,
                "error": error_msg
            }

    def add_required_columns_to_processed_files(self, symbol: str, interval: str, exchange: str = "binance") -> Dict[str, Any]:
        """Add required columns (exchange, timeframe) to processed data files.

        Args:
            symbol: Trading symbol
            interval: Data interval
            exchange: Exchange name

        Returns:
            Dictionary with column addition results
        """
        try:
            tprint_info(f"📦 Adding required columns to processed {symbol} {interval} data")

            # Get processed data directory
            data_path = Path(self.data_dir) / "binance" / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                error_msg = "No processed data directory found"
                tprint_error(f"❌ {error_msg}")
                return {"files_processed": 0, "columns_added": 0, "success": False, "error": error_msg}

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                error_msg = "No parquet files found"
                tprint_error(f"❌ {error_msg}")
                return {"files_processed": 0, "columns_added": 0, "success": False, "error": error_msg}

            files_processed = 0
            columns_added_total = 0

            for file_path in parquet_files:
                try:
                    # Read the file
                    df = pd.read_parquet(file_path)

                    # Check which columns need to be added
                    columns_to_add = {}
                    if 'exchange' not in df.columns:
                        columns_to_add['exchange'] = exchange
                    if 'timeframe' not in df.columns:
                        columns_to_add['timeframe'] = interval

                    # Ensure timestamp is available as a column (not just index)
                    if 'timestamp' not in df.columns:
                        if hasattr(df.index, 'name') and df.index.name == 'timestamp':
                            # Reset index to make timestamp a column
                            df = df.reset_index()
                        elif df.index.name is None and df.index.dtype in ['int64', 'datetime64[ns]']:
                            # Index appears to be timestamp but not named
                            df.index.name = 'timestamp'
                            df = df.reset_index()
                        else:
                            # Try to infer timestamp from index
                            df['timestamp'] = df.index

                    if columns_to_add:
                        # Add the required columns
                        for col_name, col_value in columns_to_add.items():
                            df[col_name] = col_value

                        # Save the modified file
                        df.to_parquet(file_path, index=True, compression='snappy')

                        columns_added = len(columns_to_add)
                        columns_added_total += columns_added

                        tprint(f"✅ Added {columns_added} columns to {file_path.name}", "DEBUG")
                    else:
                        tprint(f"ℹ️ Required columns already present in {file_path.name}", "DEBUG")

                    files_processed += 1

                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {e}"
                    tprint_error(f"❌ {error_msg}")
                    return {"files_processed": 0, "columns_added": 0, "success": False, "error": error_msg}

            result = {
                "files_processed": files_processed,
                "columns_added": columns_added_total,
                "columns_targeted": ['exchange', 'timeframe'],
                "success": True,
                "message": f"Processed {files_processed} files, added {columns_added_total} column instances"
            }

            tprint_success(f"✅ Column addition completed: {result['message']}")
            return result

        except Exception as e:
            error_msg = f"Column addition failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {
                "files_processed": 0,
                "columns_added": 0,
                "success": False,
                "error": error_msg
            }

    async def handle_gaps_with_column_removal(
        self,
        symbol: str,
        interval: str,
        max_gap_minutes: int,
        exchange_interface: Optional[ExchangeInterface],
        api_key: str,
        api_secret: str
    ) -> Dict[str, Any]:
        """Detect gaps and fill them, removing unwanted columns from new data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            max_gap_minutes: Maximum allowed gap in minutes
            exchange_interface: ExchangeInterface instance (preferred over api_key/api_secret)
            api_key: Exchange API key (required if exchange_interface not provided)
            api_secret: Exchange API secret (required if exchange_interface not provided)

        Returns:
            Dictionary with gap handling results
        """
        try:
            tprint_info(f"🔍 Detecting gaps in {symbol} {interval} data (max_gap: {max_gap_minutes}m)")

            # Detect gaps
            gaps = self.gap_detector.detect_gaps(symbol, interval, max_gap_minutes)

            if not gaps:
                tprint_success("✅ No gaps detected")
                return {"gaps_detected": 0, "gaps_filled": 0, "message": "No gaps to fill"}

            tprint_warning(f"⚠️ Found {len(gaps)} gaps > {max_gap_minutes} minutes")

            # Fill gaps
            gap_fill_results = await self.gap_detector.fill_gaps(gaps, exchange_interface, api_key, api_secret)

            # Remove unwanted columns and add required columns from newly downloaded data
            if gap_fill_results.get("filled_gaps", 0) > 0:
                tprint_info("🧹 Removing unwanted columns from gap-filled data")
                column_removal_results = self.remove_unwanted_columns(symbol, interval)
                if not column_removal_results.get("success", False):
                    error_msg = f"Column removal failed after gap filling: {column_removal_results.get('error')}"
                    tprint_error(f"❌ {error_msg}")
                    return {"gaps_detected": len(gaps), "gaps_filled": 0, "error": error_msg}
                gap_fill_results["column_removal"] = column_removal_results

                tprint_info("📦 Adding required columns to gap-filled data")
                column_addition_results = self.add_required_columns_to_processed_files(symbol, interval, "binance")
                if not column_addition_results.get("success", False):
                    error_msg = f"Column addition failed after gap filling: {column_addition_results.get('error')}"
                    tprint_error(f"❌ {error_msg}")
                    return {"gaps_detected": len(gaps), "gaps_filled": 0, "error": error_msg}
                gap_fill_results["column_addition"] = column_addition_results

            return gap_fill_results

        except Exception as e:
            error_msg = f"Gap handling failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {"gaps_detected": 0, "gaps_filled": 0, "error": error_msg}

    def handle_duplicates(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Analyze and handle duplicate timestamps.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Dictionary with duplicate handling results
        """
        try:
            tprint_info(f"🔍 Analyzing duplicates in {symbol} {interval} data")

            # Get data directory
            data_path = Path(self.data_dir) / "binance" / symbol.lower() / "raw" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                error_msg = "No data directory found"
                tprint_error(f"❌ {error_msg}")
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "success": False, "error": error_msg}

            # Find sample files
            parquet_files = list(data_path.glob("*.parquet"))[:5]  # Analyze up to 5 files

            if not parquet_files:
                error_msg = "No parquet files found"
                tprint_error(f"❌ {error_msg}")
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "success": False, "error": error_msg}

            # Combine data from sample files
            combined_data: List[pd.DataFrame] = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    combined_data.append(df)
                except Exception as e:
                    tprint_warning(f"⚠️ Could not read {file_path.name}: {e}")

            if not combined_data:
                error_msg = "Could not read any files"
                tprint_error(f"❌ {error_msg}")
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "success": False, "error": error_msg}

            # Concatenate all data
            full_df = pd.concat(combined_data, ignore_index=True)
            full_df = full_df.drop_duplicates()  # Remove any concatenation duplicates

            # Analyze duplicates
            analysis_result = self.duplicate_analyzer.analyze_duplicates(full_df)

            warnings: List[str] = []
            true_duplicates_removed = 0

            # Handle true duplicates (remove them)
            if analysis_result.true_duplicate_groups > 0:
                tprint_info(f"🧹 Removing {analysis_result.true_duplicate_groups} groups of true duplicates")

                for group in analysis_result.duplicate_groups:
                    if group.duplicate_type == 'true_duplicates':
                        # Remove true duplicates (keep first occurrence)
                        full_df = full_df.drop_duplicates(subset=['timestamp'], keep='first')
                        true_duplicates_removed += group.record_count - 1

            # Warn about false duplicates
            if analysis_result.false_duplicate_groups > 0:
                warning_msg = f"⚠️ Found {analysis_result.false_duplicate_groups} groups of false duplicates (same timestamp, different values) - requires manual review"
                warnings.append(warning_msg)
                tprint_warning(warning_msg)

            # Warn about mixed duplicates
            if analysis_result.mixed_duplicate_groups > 0:
                warning_msg = f"⚠️ Found {analysis_result.mixed_duplicate_groups} groups of mixed duplicates - requires detailed analysis"
                warnings.append(warning_msg)
                tprint_warning(warning_msg)

            result = {
                "files_analyzed": len(parquet_files),
                "total_records": len(full_df),
                "duplicates_found": analysis_result.total_duplicates,
                "true_duplicates_removed": true_duplicates_removed,
                "false_duplicates": analysis_result.false_duplicate_groups,
                "mixed_duplicates": analysis_result.mixed_duplicate_groups,
                "warnings": warnings,
                "recommendations": analysis_result.recommendations,
                "success": True
            }

            tprint_success(f"✅ Duplicate analysis completed: {analysis_result.total_duplicates} duplicates found, {true_duplicates_removed} true duplicates removed")
            return result

        except Exception as e:
            error_msg = f"Duplicate handling failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {
                "files_analyzed": 0,
                "duplicates_found": 0,
                "warnings": [],
                "success": False,
                "error": error_msg
            }


class KlinesDataQualityChecker:
    """Comprehensive data quality checker for klines data processing pipeline."""

    def __init__(self, data_dir: str = "historical_data") -> None:
        """Initialize the data quality checker.

        Args:
            data_dir: Base directory for historical data
        """
        self.data_dir = Path(data_dir)
        # Removed logger - using tprint instead
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer()

    def check_processed_data_quality(self, symbol: str = "ETHUSDT",
                                   intervals: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive quality checks on processed data.

        Args:
            symbol: Trading symbol to check
            intervals: List of intervals to check, defaults to all available

        Returns:
            Dictionary with quality check results
        """
        if intervals is None:
            intervals = ['1m', '5m', '15m', '30m', '1h']

        results = {
            "symbol": symbol,
            "intervals_checked": intervals,
            "overall_quality": True,
            "interval_results": {},
            "summary": {},
            "issues": [],
            "recommendations": []
        }

        tprint_info(f"🔍 Starting comprehensive data quality check for {symbol}")

        for interval in intervals:
            try:
                interval_result = self._check_single_interval_quality(symbol, interval)
                results["interval_results"][interval] = interval_result

                if not interval_result["quality_passed"]:
                    results["overall_quality"] = False
                    results["issues"].extend(interval_result["issues"])

            except Exception as e:
                tprint_error(f"❌ Error checking {interval} data: {e}")
                results["interval_results"][interval] = {
                    "quality_passed": False,
                    "issues": [f"Check failed: {str(e)}"],
                    "error": str(e)
                }
                results["overall_quality"] = False
                results["issues"].append(f"{interval}: {str(e)}")

        # Generate summary
        results["summary"] = self._generate_quality_summary(results)

        tprint_success("✅ Data quality check completed")
        return results

    def _check_single_interval_quality(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Check quality for a single interval.

        Args:
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Quality check results for this interval
        """
        result = {
            "quality_passed": True,
            "issues": [],
            "record_count": 0,
            "column_count": 0,
            "columns": [],
            "data_types": {},
            "null_summary": {},
            "feature_summary": {},
            "date_range": None,
            "statistics": {}
        }

        try:
            # Find data files for this interval
            interval_path = self.data_dir / "binance" / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not interval_path.exists():
                result["issues"].append(f"Processed data directory not found: {interval_path}")
                result["quality_passed"] = False
                return result

            # Sample a few files to check quality
            sample_files = self._get_sample_files(interval_path, max_files=3)

            if not sample_files:
                result["issues"].append("No parquet files found")
                result["quality_passed"] = False
                return result

            # Read and analyze sample data
            total_records = 0
            combined_dtypes: Dict[str, str] = {}
            combined_nulls: Dict[str, int] = {}
            date_ranges: List[datetime] = []

            for file_path in sample_files:
                try:
                    df = pd.read_parquet(file_path)

                    # Accumulate statistics
                    total_records += len(df)
                    date_ranges.extend([df.index.min(), df.index.max()])

                    # Check data types consistency
                    for col, dtype in df.dtypes.items():
                        if col not in combined_dtypes:
                            combined_dtypes[col] = str(dtype)
                        elif combined_dtypes[col] != str(dtype):
                            result["issues"].append(f"Inconsistent dtype for {col}: {combined_dtypes[col]} vs {str(dtype)}")

                    # Check for null values
                    null_counts = df.isnull().sum()
                    for col, count in null_counts.items():
                        if col not in combined_nulls:
                            combined_nulls[col] = count
                        else:
                            combined_nulls[col] += count

                except Exception as e:
                    result["issues"].append(f"Error reading {file_path.name}: {e}")

            # Store results
            result["record_count"] = total_records
            result["column_count"] = len(combined_dtypes)
            result["columns"] = list(combined_dtypes.keys())
            result["data_types"] = combined_dtypes
            result["null_summary"] = combined_nulls

            if date_ranges:
                result["date_range"] = {
                    "start": min(date_ranges),
                    "end": max(date_ranges)
                }

            # Validate data quality
            quality_issues = self._validate_data_quality(combined_dtypes, combined_nulls, total_records)
            result["issues"].extend(quality_issues)

            # Check features
            result["feature_summary"] = self._analyze_features(combined_dtypes)

            # Check for duplicate timestamps
            result["duplicate_analysis"] = self._check_duplicate_timestamps(sample_files)

            # Generate statistics
            result["statistics"] = self._generate_statistics(result)

            if result["issues"]:
                result["quality_passed"] = False

        except Exception as e:
            result["issues"].append(f"Quality check failed: {e}")
            result["quality_passed"] = False

        return result

    def _get_sample_files(self, interval_path: Path, max_files: int = 3) -> List[Path]:
        """Get a sample of parquet files for quality checking.

        Args:
            interval_path: Path to interval directory
            max_files: Maximum number of files to sample

        Returns:
            List of parquet file paths
        """
        all_files = []

        # Recursively find all parquet files
        for parquet_file in interval_path.rglob("*.parquet"):
            all_files.append(parquet_file)

        # Sort by modification time and take most recent
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return all_files[:max_files]

    def _check_duplicate_timestamps(self, sample_files: List[Path]) -> Dict[str, Any]:
        """Check for duplicate timestamps in sample files using comprehensive analysis.

        Args:
            sample_files: List of parquet files to analyze

        Returns:
            Dictionary with duplicate analysis results
        """
        duplicate_results = {
            "total_files_analyzed": len(sample_files),
            "files_with_duplicates": 0,
            "total_duplicate_records": 0,
            "duplicate_groups": 0,
            "true_duplicates": 0,
            "false_duplicates": 0,
            "mixed_duplicates": 0,
            "duplicate_issues": [],
            "recommendations": [],
            "detailed_analysis": {}
        }

        try:
            # Combine data from all sample files for analysis
            combined_data: List[pd.DataFrame] = []
            for file_path in sample_files:
                try:
                    df = pd.read_parquet(file_path)
                    combined_data.append(df)
                except Exception as e:
                    duplicate_results["duplicate_issues"].append(f"Error reading {file_path.name}: {e}")

            if not combined_data:
                duplicate_results["duplicate_issues"].append("No files could be read for duplicate analysis")
                return duplicate_results

            # Concatenate all data
            full_df = pd.concat(combined_data, ignore_index=True)

            # Remove any existing duplicates from concatenation
            full_df = full_df.drop_duplicates()

            # Ensure timestamp is available as a column for duplicate analysis
            if 'timestamp' not in full_df.columns:
                if hasattr(full_df.index, 'name') and full_df.index.name == 'timestamp':
                    full_df = full_df.reset_index()
                elif full_df.index.name is None and full_df.index.dtype in ['int64', 'datetime64[ns]', 'datetime64[ns, UTC]']:
                    full_df.index.name = 'timestamp'
                    full_df = full_df.reset_index()
                else:
                    # Try to infer timestamp from index
                    full_df['timestamp'] = full_df.index
                    full_df.index.name = 'timestamp'
                    full_df = full_df.reset_index(drop=True)

            # Analyze duplicates
            analysis_result = self.duplicate_analyzer.analyze_duplicates(full_df)

            # Populate results
            duplicate_results.update({
                "total_duplicate_records": analysis_result.total_duplicates,
                "duplicate_groups": len(analysis_result.duplicate_groups),
                "true_duplicates": analysis_result.true_duplicate_groups,
                "false_duplicates": analysis_result.false_duplicate_groups,
                "mixed_duplicates": analysis_result.mixed_duplicate_groups,
                "recommendations": analysis_result.recommendations,
                "detailed_analysis": {
                    "summary_stats": analysis_result.summary_stats,
                    "duplicate_type_distribution": analysis_result.summary_stats.get("duplicate_type_distribution", {})
                }
            })

            # Check if any files have duplicates
            if analysis_result.total_duplicates > 0:
                duplicate_results["files_with_duplicates"] = len(sample_files)  # Assume all files contribute

                # Add issues based on duplicate types
                if analysis_result.false_duplicate_groups > 0:
                    duplicate_results["duplicate_issues"].append(
                        f"Found {analysis_result.false_duplicate_groups} groups of false duplicates "
                        "(same timestamp, different values) - requires investigation"
                    )

                if analysis_result.true_duplicate_groups > 0:
                    duplicate_results["duplicate_issues"].append(
                        f"Found {analysis_result.true_duplicate_groups} groups of true duplicates "
                        "(identical records) - safe to remove"
                    )

                if analysis_result.mixed_duplicate_groups > 0:
                    duplicate_results["duplicate_issues"].append(
                        f"Found {analysis_result.mixed_duplicate_groups} groups of mixed duplicates "
                        "- requires detailed analysis"
                    )

            tprint_info(f"✅ Duplicate analysis completed: {analysis_result.total_duplicates} duplicates in {len(analysis_result.duplicate_groups)} groups")

        except Exception as e:
            tprint_error(f"Error in duplicate timestamp analysis: {e}")
            duplicate_results["duplicate_issues"].append(f"Analysis failed: {e}")

        return duplicate_results

    def _validate_data_quality(self, dtypes: Dict[str, str],
                             null_counts: Dict[str, int],
                             total_records: int) -> List[str]:
        """Validate data quality requirements.

        Args:
            dtypes: Column data types
            null_counts: Null value counts per column
            total_records: Total number of records

        Returns:
            List of quality issues found
        """
        issues = []

        # Required columns check
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in dtypes]
        if missing_cols:
            issues.append(f"Missing required OHLCV columns: {missing_cols}")

        # Data type validation
        expected_numeric = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
        for col in expected_numeric:
            if col in dtypes:
                dtype_str = dtypes[col]
                if not any(t in dtype_str for t in ['float', 'int']):
                    issues.append(f"Column {col} should be numeric, got {dtype_str}")

        # Null value validation
        max_null_pct = 0.01  # 1% maximum nulls
        for col, null_count in null_counts.items():
            if null_count > 0:
                null_pct = null_count / total_records
                if null_pct > max_null_pct:
                    issues.append(f"Column {col} has {null_pct:.1%} null values")
                elif null_count > 0:
                    issues.append(f"Column {col} has {null_count} null values")

        # Timestamp validation
        if 'timestamp' in dtypes:
            dtype_str = dtypes['timestamp']
            if not any(t in dtype_str for t in ['int', 'datetime']):
                issues.append(f"Timestamp column should be int64 or datetime, got {dtype_str}")

        return issues

    def _analyze_features(self, dtypes: Dict[str, str]) -> Dict[str, Any]:
        """Analyze feature completeness.

        Args:
            dtypes: Column data types

        Returns:
            Feature analysis summary
        """
        feature_analysis = {
            "return_features": [],
            "technical_features": [],
            "time_features": [],
            "total_features": 0
        }

        # Categorize features
        for col in dtypes.keys():
            if 'return' in col.lower() or 'log' in col.lower():
                feature_analysis["return_features"].append(col)
            elif any(x in col.lower() for x in ['range', 'body', 'ma', 'rsi', 'macd', 'bollinger']):
                feature_analysis["technical_features"].append(col)
            elif any(x in col.lower() for x in ['hour', 'day', 'week', 'month', 'time']):
                feature_analysis["time_features"].append(col)

        feature_analysis["total_features"] = (
            len(feature_analysis["return_features"]) +
            len(feature_analysis["technical_features"]) +
            len(feature_analysis["time_features"])
        )

        return feature_analysis

    def _generate_statistics(self, interval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary.

        Args:
            interval_result: Results for a single interval

        Returns:
            Statistical summary
        """
        stats = {
            "total_records": interval_result["record_count"],
            "total_columns": interval_result["column_count"],
            "null_percentage": 0.0,
            "features_count": interval_result["feature_summary"]["total_features"]
        }

        # Calculate null percentage
        total_nulls = sum(interval_result["null_summary"].values())
        if interval_result["record_count"] > 0:
            stats["null_percentage"] = (total_nulls / interval_result["record_count"]) * 100

        return stats

    def _generate_quality_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall quality summary.

        Args:
            results: Complete quality check results

        Returns:
            Summary statistics
        """
        summary = {
            "total_intervals": len(results["intervals_checked"]),
            "passed_intervals": 0,
            "failed_intervals": 0,
            "total_issues": len(results["issues"]),
            "total_records": 0,
            "avg_null_percentage": 0.0
        }

        null_percentages = []

        for interval_result in results["interval_results"].values():
            if interval_result.get("quality_passed", False):
                summary["passed_intervals"] += 1
            else:
                summary["failed_intervals"] += 1

            summary["total_records"] += interval_result.get("record_count", 0)

            if "statistics" in interval_result and "null_percentage" in interval_result["statistics"]:
                null_percentages.append(interval_result["statistics"]["null_percentage"])

        if null_percentages:
            summary["avg_null_percentage"] = sum(null_percentages) / len(null_percentages)

        return summary

    def print_quality_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted quality report.

        Args:
            results: Quality check results
        """
        print("\n" + "="*60)
        print("📊 KLINES DATA QUALITY REPORT")
        print("="*60)
        print(f"Symbol: {results['symbol']}")
        print(f"Intervals Checked: {', '.join(results['intervals_checked'])}")
        print(f"Overall Quality: {'✅ PASSED' if results['overall_quality'] else '❌ FAILED'}")
        print()

        # Summary
        summary = results['summary']
        print("📈 SUMMARY:")
        print(f"  Total Intervals: {summary['total_intervals']}")
        print(f"  Passed: {summary['passed_intervals']}")
        print(f"  Failed: {summary['failed_intervals']}")
        print(f"  Total Issues: {summary['total_issues']}")
        print(f"  Total Records: {summary['total_records']:,}")
        print(f"  Avg Null %: {summary['avg_null_percentage']:.2f}")
        print()

        # Interval details
        print("📋 INTERVAL DETAILS:")
        for interval, interval_result in results['interval_results'].items():
            status = "✅" if interval_result.get('quality_passed', False) else "❌"
            record_count = interval_result.get('record_count', 0)
            issue_count = len(interval_result.get('issues', []))
            duplicate_analysis = interval_result.get('duplicate_analysis', {})

            print(f"  {interval.upper()}: {status} {record_count:,} records, {issue_count} issues")

            # Show duplicate analysis summary
            if duplicate_analysis.get('total_duplicate_records', 0) > 0:
                dup_count = duplicate_analysis['total_duplicate_records']
                dup_groups = duplicate_analysis['duplicate_groups']
                false_dup = duplicate_analysis.get('false_duplicates', 0)
                true_dup = duplicate_analysis.get('true_duplicates', 0)

                print(f"    📊 Duplicates: {dup_count} records in {dup_groups} groups")
                print(f"       True duplicates: {true_dup}, False duplicates: {false_dup}")

                if duplicate_analysis.get('duplicate_issues'):
                    for issue in duplicate_analysis['duplicate_issues'][:2]:
                        print(f"       ⚠️  {issue}")

            if interval_result.get('issues'):
                for issue in interval_result['issues'][:3]:  # Show first 3 issues
                    print(f"    • {issue}")
                if len(interval_result['issues']) > 3:
                    print(f"    • ... and {len(interval_result['issues']) - 3} more issues")
        print()

        # Recommendations
        all_recommendations = []

        # Collect recommendations from issues
        if results['issues']:
            all_recommendations.extend(results['issues'])

        # Collect duplicate-specific recommendations
        for interval_result in results['interval_results'].values():
            duplicate_analysis = interval_result.get('duplicate_analysis', {})
            if duplicate_analysis.get('recommendations'):
                all_recommendations.extend(duplicate_analysis['recommendations'])

        if all_recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in all_recommendations[:5]:
                print(f"  • {rec}")
            if len(all_recommendations) > 5:
                print(f"  • ... and {len(all_recommendations) - 5} more recommendations")

        print("="*60)


# Convenience functions
async def run_enhanced_klines_pipeline(
    symbol: str = "ETHUSDT",
    years: Optional[int] = None,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    interval: str = "1m",
    max_gap_minutes: int = 1,
    create_consolidated: bool = True
) -> Dict[str, Any]:
    """Run the enhanced klines processing pipeline.

    This function:
    - Downloads data for the specified symbol using HistoricalDataPipeline
    - Removes taker_buy_base, taker_buy_quote, year columns
    - Detects gaps > 1m and re-downloads if needed (with column removal)
    - Analyzes duplicates (warns on false duplicates, removes true duplicates)
    - Runs final quality checks
    - Creates consolidated features file with required columns: ['timestamp', 'exchange', 'timeframe']
    - Uses fast fail pattern with no fallbacks or mock data
    - Integrates data standardizer for consistent format

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        years: Number of years of data to download (default: from centralized config)
        data_dir: Base directory for data storage
        api_key: Binance API key
        api_secret: Binance API secret
        interval: Kline interval (default: "1m")
        max_gap_minutes: Maximum allowed gap in minutes (default: 1)
        create_consolidated: Whether to create consolidated features file (default: True)

    Returns:
        Dictionary with complete pipeline results
    """
    # Use centralized configuration if years not specified
    if years is None:
        from src.config.pipeline_modes import get_full_mode_config
        mode_config = get_full_mode_config()
        years = mode_config.lookback_years
    
    pipeline = KlinesDataProcessingPipeline(data_dir)

    print(f"🚀 Starting enhanced {symbol} {interval} data pipeline ({years} years)")
    print(f"📁 Data directory: {data_dir}")
    print(f"⏱️  Interval: {interval}")
    print(f"🎯 Max gap threshold: {max_gap_minutes} minutes")
    print()

    results = await pipeline.run_complete_pipeline(
        symbol=symbol,
        years=years,
        interval=interval,
        api_key=api_key,
        api_secret=api_secret,
        max_gap_minutes=max_gap_minutes,
        create_consolidated=create_consolidated
    )

    # Print summary
    print("\n" + "="*80)
    print("📊 ENHANCED PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Symbol: {results['symbol']}")
    print(f"Years: {results['years']}")
    print(f"Interval: {results['interval']}")
    print(f"Pipeline Success: {'✅ YES' if results['pipeline_success'] else '❌ NO'}")
    print(f"Steps Completed: {len(results['steps_completed'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print()

    if results['steps_completed']:
        print("✅ COMPLETED STEPS:")
        for step in results['steps_completed']:
            print(f"  • {step}")

    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  • {error}")
        if len(results['errors']) > 5:
            print(f"  • ... and {len(results['errors']) - 5} more errors")

    if results['warnings']:
        print("\n⚠️ WARNINGS:")
        for warning in results['warnings'][:5]:  # Show first 5 warnings
            print(f"  • {warning}")
        if len(results['warnings']) > 5:
            print(f"  • ... and {len(results['warnings']) - 5} more warnings")

    print("\n" + "="*80)

    return results


if __name__ == "__main__":
    # Example usage - download klines data with configurable symbol and years
    async def main():
        # Parse command line arguments for symbol and years
        symbol = "ETHUSDT"  # default
        years = None  # Will use centralized config
        
        # Check for additional command line arguments
        if len(sys.argv) > 1:
            # Try to parse symbol from command line
            potential_symbol = sys.argv[1].upper()
            if potential_symbol not in ["TEST_CONSOLIDATED", "TEST_COLUMNS"]:
                symbol = potential_symbol
                
            # Try to parse years from command line
            if len(sys.argv) > 2:
                try:
                    years = int(sys.argv[2])
                except ValueError:
                    print(f"⚠️ Invalid years argument '{sys.argv[2]}', using default: {years}")
        
        print(f"Starting enhanced {symbol} {years}-year 1m klines data download pipeline...")

        # Run the enhanced pipeline
        results = await run_enhanced_klines_pipeline(symbol=symbol, years=years)

        print(f"\nPipeline completed with success: {results.get('pipeline_success', False)}")

        # Print any warnings or errors
        if results.get('warnings'):
            print("\n⚠️ WARNINGS:")
            for warning in results['warnings']:
                print(f"  • {warning}")

        if results.get('errors'):
            print("\n❌ ERRORS:")
            for error in results['errors']:
                print(f"  • {error}")

    # Run the main function
    asyncio.run(main())