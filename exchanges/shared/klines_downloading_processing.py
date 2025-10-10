"""
Klines Data Downloading and Processing Pipeline

This module provides a complete pipeline for downloading, processing, and quality-checking
historical klines data from any exchange with gap detection, duplicate handling, and column management.

Features:
- Download historical klines data using HistoricalDataPipeline
- Detect and fill gaps > 1m
- Identify and handle duplicate timestamps (warn on false duplicates, remove true duplicates)
- Remove unwanted columns (taker_buy_base, taker_buy_quote, year)
- Comprehensive data quality checks
- Exchange-agnostic data standardization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import asyncio
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger
from src.utils.data.quality.comprehensive_duplicate_analyzer import (
    ComprehensiveDuplicateAnalyzer,
    analyze_duplicates_comprehensive
)
from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
from src.utils.data.klines_parquet import KlinesParquetManager
from exchanges.shared import ExchangeDataStandardizer
from src.utils.data.gap_detector import GapDetector


class KlinesDataProcessingPipeline:
    """Complete pipeline for downloading, processing, and quality-checking klines data from any exchange."""

    def __init__(self, exchange: str, data_dir: str = "historical_data"):
        """Initialize the processing pipeline.

        Args:
            exchange: Exchange name (binance, bingx, mexc, okx, gateio, phemex, etc.)
            data_dir: Base directory for historical data
        """
        self.exchange = exchange.lower()
        self.data_dir = data_dir
        self.logger = system_logger.getChild(f"KlinesDataProcessingPipeline-{self.exchange.upper()}")

        # Initialize components
        self.historical_pipeline = HistoricalDataPipeline(data_dir)
        self.gap_detector = GapDetector(data_dir)
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer(self.logger)

        # Initialize standardized data managers
        self.parquet_manager = KlinesParquetManager(data_dir, self.exchange)
        self.data_standardizer = ExchangeDataStandardizer(data_dir)

        # Quality checker will be initialized when first used
        self._quality_checker = None

        # Columns to remove (exchange-agnostic)
        self.columns_to_remove = ['taker_buy_base', 'taker_buy_quote', 'year']

    def standardize_data_format(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Standardize data format using centralized standardizer.
        
        Args:
            df: Raw DataFrame from exchange
            symbol: Trading symbol
            interval: Data interval
            
        Returns:
            Standardized DataFrame
        """
        try:
            standardized_df, report = self.data_standardizer.standardize_data(
                df, self.exchange, symbol, interval, validate_quality=True
            )
            
            if report['success']:
                self.logger.info(f"✅ Data standardized: {len(standardized_df)} records for {symbol} {interval}")
                if report.get('warnings'):
                    for warning in report['warnings']:
                        self.logger.warning(f"⚠️ {warning}")
            else:
                self.logger.error(f"❌ Data standardization failed: {report.get('errors', [])}")
                
            return standardized_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to standardize data format: {e}")
            return df

    def save_standardized_data(self, df: pd.DataFrame, symbol: str, interval: str, data_type: str = "raw") -> bool:
        """Save data using standardized KlinesParquetManager.
        
        Args:
            df: DataFrame to save
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.parquet_manager.write_data(df, symbol, interval, data_type)
        except Exception as e:
            self.logger.error(f"❌ Failed to save standardized data: {e}")
            return False

    def load_standardized_data(self, symbol: str, interval: str, data_type: str = "raw", 
                             start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Load data using standardized KlinesParquetManager.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with data or None if not found
        """
        try:
            return self.parquet_manager.read_data(symbol, interval, start_date, end_date, data_type)
        except Exception as e:
            self.logger.error(f"❌ Failed to load standardized data: {e}")
            return None

    def validate_data_quality(self, df: pd.DataFrame, context: str = "") -> Dict[str, Any]:
        """Validate data quality using centralized standardizer.
        
        Args:
            df: DataFrame to validate
            context: Context for validation
            
        Returns:
            Validation results
        """
        try:
            # Use the standardizer's quality validation
            _, report = self.data_standardizer.standardize_data(
                df, self.exchange, "VALIDATION", "1m", validate_quality=True
            )
            
            quality_info = report.get('quality_validation', {})
            return {
                'passed': quality_info.get('passed', False),
                'quality_score': quality_info.get('quality_score', 0.0),
                'issues': quality_info.get('issues', []),
                'warnings': quality_info.get('warnings', []),
                'metrics': quality_info.get('metrics', {}),
                'standardization_report': report
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to validate data quality: {e}")
            return {'passed': False, 'error': str(e)}

    def process_klines_data(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        interval: str, 
        save_data: bool = True
    ) -> pd.DataFrame:
        """Process klines data using the shared pipeline.
        
        Args:
            df: Raw DataFrame from exchange
            symbol: Trading symbol
            interval: Data interval
            save_data: Whether to save processed data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Standardize data format
            standardized_df = self.standardize_data_format(df, symbol, interval)
            
            if save_data:
                # Save standardized data
                self.save_standardized_data(standardized_df, symbol, interval, "raw")
            
            return standardized_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process klines data: {e}")
            return df

    def get_processed_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get previously processed data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Processed DataFrame or None
        """
        return self.load_standardized_data(symbol, interval, "raw", start_date, end_date)

    @property
    def quality_checker(self):
        """Lazy initialization of quality checker."""
        if self._quality_checker is None:
            self._quality_checker = ExchangeKlinesDataQualityChecker(self.exchange, self.data_dir)
        return self._quality_checker

    def create_consolidated_features_file(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        exchange: str = None
    ) -> Dict[str, Any]:
        """Create a consolidated features parquet file with required columns.

        This creates a file with the format: historical_data/features_{EXCHANGE}_{SYMBOL}_consolidated.parquet
        containing the required columns: ['timestamp', 'exchange', 'timeframe']

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            interval: Time interval (e.g., "1m")
            exchange: Exchange name (default: uses self.exchange)

        Returns:
            Dictionary with consolidation results
        """
        try:
            if exchange is None:
                exchange = self.exchange
                
            self.logger.info(f"📦 Creating consolidated features file for {symbol} {interval}")

            # Define output file path
            output_file = Path(self.data_dir) / f"features_{exchange.lower()}_{symbol.upper()}_consolidated.parquet"

            # Find processed data files
            data_path = Path(self.data_dir) / exchange.lower() / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                return {
                    "success": False,
                    "error": f"Processed data directory not found: {data_path}",
                    "files_processed": 0,
                    "total_records": 0
                }

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                return {
                    "success": False,
                    "error": "No parquet files found in processed directory",
                    "files_processed": 0,
                    "total_records": 0
                }

            self.logger.info(f"🔍 Found {len(parquet_files)} parquet files to consolidate")

            # Combine all data
            combined_data = []
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

                    self.logger.debug(f"✅ Processed {file_path.name}: {len(df)} records")

                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path.name}: {e}")
                    continue

            if not combined_data:
                return {
                    "success": False,
                    "error": "Failed to process any data files",
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

            self.logger.info("✅ Consolidated features file created successfully")
            self.logger.info(f"   📁 Output: {output_file}")
            self.logger.info(f"   📊 Records: {len(consolidated_df):,}")
            self.logger.info(f"   🗂️  Columns: {len(consolidated_df.columns)}")
            self.logger.info(f"   📏 Size: {result['file_size_mb']} MB")

            return result

        except Exception as e:
            self.logger.exception(f"❌ Failed to create consolidated features file: {e}")
            return {
                "success": False,
                "error": str(e),
                "files_processed": 0,
                "total_records": 0
            }

    async def run_complete_pipeline(
        self,
        symbol: str = "ETHUSDT",
        years: int = None,
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
            api_key: BingX API key
            api_secret: BingX API secret
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
            
            self.logger.info(f"🚀 Starting complete klines processing pipeline for {symbol}")

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
            self.logger.info(f"📥 Step 1: Downloading {years} years of {symbol} {interval} data")
            download_results = await self.historical_pipeline.run_complete_pipeline(
                symbol=symbol,
                years=years,
                api_key=api_key,
                api_secret=api_secret,
                target_intervals=[interval] if interval != "1m" else []
            )

            if download_results.get("pipeline_success", False):
                results["steps_completed"].append("download")
                results["summary"]["download"] = download_results
                self.logger.info("✅ Data download completed")
            else:
                results["errors"].extend(download_results.get("errors", []))
                self.logger.error("❌ Data download failed")
                return results

            # Step 2: Remove unwanted columns
            self.logger.info("🧹 Step 2: Removing unwanted columns")
            column_removal_results = self.remove_unwanted_columns(symbol, interval)
            results["steps_completed"].append("column_removal")
            results["summary"]["column_removal"] = column_removal_results

            # Step 2.5: Add required columns to processed files
            self.logger.info("📦 Step 2.5: Adding required columns (exchange, timeframe) to processed files")
            column_addition_results = self.add_required_columns_to_processed_files(symbol, interval, self.exchange)
            results["steps_completed"].append("column_addition")
            results["summary"]["column_addition"] = column_addition_results

            # Step 3: Detect and fill gaps > 1m
            self.logger.info("🔍 Step 3: Detecting gaps > 1m and re-downloading if needed")
            gap_results = await self.handle_gaps_with_column_removal(
                symbol, interval, max_gap_minutes, api_key, api_secret
            )
            results["steps_completed"].append("gap_handling")
            results["summary"]["gap_handling"] = gap_results

            # Step 4: Handle duplicates (warn on false, remove true)
            self.logger.info("🔍 Step 4: Analyzing and handling duplicate timestamps")
            duplicate_results = self.handle_duplicates(symbol, interval)
            results["steps_completed"].append("duplicate_handling")
            results["summary"]["duplicate_handling"] = duplicate_results

            # Add warnings from duplicate analysis
            if duplicate_results.get("warnings"):
                results["warnings"].extend(duplicate_results["warnings"])

            # Step 5: Final quality check
            self.logger.info("✅ Step 5: Running final data quality check")
            quality_results = self.quality_checker.check_processed_data_quality(
                symbol, [interval]
            )
            results["steps_completed"].append("quality_check")
            results["summary"]["quality_check"] = quality_results

            # Step 6: Create consolidated features file (optional)
            if create_consolidated:
                self.logger.info("📦 Step 6: Creating consolidated features file")
                consolidated_results = self.create_consolidated_features_file(
                    symbol=symbol,
                    interval=interval,
                    exchange=self.exchange
                )
                results["steps_completed"].append("consolidated_file_creation")
                results["summary"]["consolidated_file_creation"] = consolidated_results

                if not consolidated_results.get("success", False):
                    results["warnings"].append(f"Consolidated file creation failed: {consolidated_results.get('error', 'Unknown error')}")
                else:
                    self.logger.info("✅ Consolidated features file created successfully")

            # Overall success
            results["pipeline_success"] = len(results["errors"]) == 0
            results["completion_time"] = datetime.now().isoformat()

            self.logger.info(f"🎉 Pipeline completed: {len(results['steps_completed'])} steps, {len(results['errors'])} errors, {len(results['warnings'])} warnings")

            return results

        except Exception as e:
            self.logger.exception(f"❌ Pipeline failed: {e}")
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
            self.logger.info(f"🧹 Removing columns {self.columns_to_remove} from {symbol} {interval} data")

            # Get data directory
            data_path = Path(self.data_dir) / self.exchange / symbol.lower() / "raw" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                return {"files_processed": 0, "columns_removed": 0, "message": "No data directory found"}

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                return {"files_processed": 0, "columns_removed": 0, "message": "No parquet files found"}

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

                        self.logger.debug(f"✅ Removed {columns_removed} columns from {file_path.name}")
                    else:
                        self.logger.debug(f"ℹ️ No unwanted columns found in {file_path.name}")

                    files_processed += 1

                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path.name}: {e}")

            result = {
                "files_processed": files_processed,
                "columns_removed": columns_removed_total,
                "columns_targeted": self.columns_to_remove,
                "success": True,
                "message": f"Processed {files_processed} files, removed {columns_removed_total} column instances"
            }

            self.logger.info(f"✅ Column removal completed: {result['message']}")
            return result

        except Exception as e:
            self.logger.exception(f"❌ Column removal failed: {e}")
            return {
                "files_processed": 0,
                "columns_removed": 0,
                "success": False,
                "error": str(e)
            }

    def add_required_columns_to_processed_files(self, symbol: str, interval: str, exchange: str = "bingx") -> Dict[str, Any]:
        """Add required columns (exchange, timeframe) to processed data files.

        Args:
            symbol: Trading symbol
            interval: Data interval
            exchange: Exchange name

        Returns:
            Dictionary with column addition results
        """
        try:
            self.logger.info(f"📦 Adding required columns to processed {symbol} {interval} data")

            # Get processed data directory
            data_path = Path(self.data_dir) / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                return {"files_processed": 0, "columns_added": 0, "message": "No processed data directory found"}

            # Find all parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if not parquet_files:
                return {"files_processed": 0, "columns_added": 0, "message": "No parquet files found"}

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

                        self.logger.debug(f"✅ Added {columns_added} columns to {file_path.name}")
                    else:
                        self.logger.debug(f"ℹ️ Required columns already present in {file_path.name}")

                    files_processed += 1

                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path.name}: {e}")

            result = {
                "files_processed": files_processed,
                "columns_added": columns_added_total,
                "columns_targeted": ['exchange', 'timeframe'],
                "success": True,
                "message": f"Processed {files_processed} files, added {columns_added_total} column instances"
            }

            self.logger.info(f"✅ Column addition completed: {result['message']}")
            return result

        except Exception as e:
            self.logger.exception(f"❌ Column addition failed: {e}")
            return {
                "files_processed": 0,
                "columns_added": 0,
                "success": False,
                "error": str(e)
            }

    async def handle_gaps_with_column_removal(
        self,
        symbol: str,
        interval: str,
        max_gap_minutes: int,
        api_key: str,
        api_secret: str
    ) -> Dict[str, Any]:
        """Detect gaps and fill them, removing unwanted columns from new data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            max_gap_minutes: Maximum allowed gap in minutes
            api_key: BingX API key
            api_secret: BingX API secret

        Returns:
            Dictionary with gap handling results
        """
        try:
            self.logger.info(f"🔍 Detecting gaps in {symbol} {interval} data (max_gap: {max_gap_minutes}m)")

            # Detect gaps
            gaps = self.gap_detector.detect_gaps(symbol, interval, max_gap_minutes)

            if not gaps:
                self.logger.info("✅ No gaps detected")
                return {"gaps_detected": 0, "gaps_filled": 0, "message": "No gaps to fill"}

            self.logger.info(f"⚠️ Found {len(gaps)} gaps > {max_gap_minutes} minutes")

            # Fill gaps
            gap_fill_results = await self.gap_detector.fill_gaps(gaps, api_key, api_secret)

            # Remove unwanted columns and add required columns from newly downloaded data
            if gap_fill_results.get("filled_gaps", 0) > 0:
                self.logger.info("🧹 Removing unwanted columns from gap-filled data")
                column_removal_results = self.remove_unwanted_columns(symbol, interval)
                gap_fill_results["column_removal"] = column_removal_results

                self.logger.info("📦 Adding required columns to gap-filled data")
                column_addition_results = self.add_required_columns_to_processed_files(symbol, interval, "bingx")
                gap_fill_results["column_addition"] = column_addition_results

            return gap_fill_results

        except Exception as e:
            self.logger.exception(f"❌ Gap handling failed: {e}")
            return {"gaps_detected": 0, "gaps_filled": 0, "error": str(e)}

    def handle_duplicates(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Analyze and handle duplicate timestamps.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Dictionary with duplicate handling results
        """
        try:
            self.logger.info(f"🔍 Analyzing duplicates in {symbol} {interval} data")

            # Get data directory
            data_path = Path(self.data_dir) / self.exchange / symbol.lower() / "raw" / f"{symbol.lower()}_{interval}"

            if not data_path.exists():
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "message": "No data directory found"}

            # Find sample files
            parquet_files = list(data_path.glob("*.parquet"))[:5]  # Analyze up to 5 files

            if not parquet_files:
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "message": "No parquet files found"}

            # Combine data from sample files
            combined_data = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    combined_data.append(df)
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read {file_path.name}: {e}")

            if not combined_data:
                return {"files_analyzed": 0, "duplicates_found": 0, "warnings": [], "message": "Could not read any files"}

            # Concatenate all data
            full_df = pd.concat(combined_data, ignore_index=True)
            full_df = full_df.drop_duplicates()  # Remove any concatenation duplicates

            # Analyze duplicates
            analysis_result = self.duplicate_analyzer.analyze_duplicates(full_df)

            warnings = []
            true_duplicates_removed = 0

            # Handle true duplicates (remove them)
            if analysis_result.true_duplicate_groups > 0:
                self.logger.info(f"🧹 Removing {analysis_result.true_duplicate_groups} groups of true duplicates")

                for group in analysis_result.duplicate_groups:
                    if group.duplicate_type == 'true_duplicates':
                        # Remove true duplicates (keep first occurrence)
                        full_df = full_df.drop_duplicates(subset=['timestamp'], keep='first')
                        true_duplicates_removed += group.record_count - 1

            # Warn about false duplicates
            if analysis_result.false_duplicate_groups > 0:
                warning_msg = f"⚠️ Found {analysis_result.false_duplicate_groups} groups of false duplicates (same timestamp, different values) - requires manual review"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)

            # Warn about mixed duplicates
            if analysis_result.mixed_duplicate_groups > 0:
                warning_msg = f"⚠️ Found {analysis_result.mixed_duplicate_groups} groups of mixed duplicates - requires detailed analysis"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)

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

            self.logger.info(f"✅ Duplicate analysis completed: {analysis_result.total_duplicates} duplicates found, {true_duplicates_removed} true duplicates removed")
            return result

        except Exception as e:
            self.logger.exception(f"❌ Duplicate handling failed: {e}")
            return {
                "files_analyzed": 0,
                "duplicates_found": 0,
                "warnings": [],
                "success": False,
                "error": str(e)
            }


class ExchangeKlinesDataQualityChecker:
    """Comprehensive data quality checker for exchange klines data processing pipeline."""

    def __init__(self, exchange: str, data_dir: str = "historical_data"):
        """Initialize the data quality checker.

        Args:
            exchange: Exchange name
            data_dir: Base directory for historical data
        """
        self.exchange = exchange.lower()
        self.data_dir = Path(data_dir)
        self.logger = system_logger.getChild(f"ExchangeKlinesDataQualityChecker-{self.exchange.upper()}")
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer(self.logger)

    def check_processed_data_quality(self, symbol: str = "ETHUSDT",
                                   intervals: List[str] = None) -> Dict[str, Any]:
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

        self.logger.info(f"🔍 Starting comprehensive data quality check for {symbol}")

        for interval in intervals:
            try:
                interval_result = self._check_single_interval_quality(symbol, interval)
                results["interval_results"][interval] = interval_result

                if not interval_result["quality_passed"]:
                    results["overall_quality"] = False
                    results["issues"].extend(interval_result["issues"])

            except Exception as e:
                self.logger.error(f"❌ Error checking {interval} data: {e}")
                results["interval_results"][interval] = {
                    "quality_passed": False,
                    "issues": [f"Check failed: {str(e)}"],
                    "error": str(e)
                }
                results["overall_quality"] = False
                results["issues"].append(f"{interval}: {str(e)}")

        # Generate summary
        results["summary"] = self._generate_quality_summary(results)

        self.logger.info("✅ Data quality check completed")
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
            interval_path = self.data_dir / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

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
            combined_dtypes = {}
            combined_nulls = {}
            date_ranges = []

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
            combined_data = []
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

            self.logger.info(f"✅ Duplicate analysis completed: {analysis_result.total_duplicates} duplicates in {len(analysis_result.duplicate_groups)} groups")

        except Exception as e:
            self.logger.error(f"Error in duplicate timestamp analysis: {e}")
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


# Convenience functions
async def run_exchange_klines_pipeline(
    exchange: str,
    symbol: str = "ETHUSDT",
    years: int = None,
    interval: str = "1m",
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    max_gap_minutes: int = 1,
    create_consolidated: bool = True
) -> Dict[str, Any]:
    """Run the complete pipeline for downloading klines data for any symbol from any exchange.

    This function:
    - Downloads data for the specified symbol using HistoricalDataPipeline
    - Removes taker_buy_base, taker_buy_quote, year columns
    - Detects gaps > 1m and re-downloads if needed (with column removal)
    - Analyzes duplicates (warns on false duplicates, removes true duplicates)
    - Runs final quality checks
    - Creates consolidated features file with required columns: ['timestamp', 'exchange', 'timeframe']

    Args:
        exchange: Exchange name (binance, bingx, mexc, okx, gateio, phemex, etc.)
        symbol: Trading symbol (default: "ETHUSDT")
        years: Number of years of data to download (default: from centralized config)
        data_dir: Base directory for data storage
        api_key: Exchange API key
        api_secret: Exchange API secret
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
    
    pipeline = KlinesDataProcessingPipeline(exchange, data_dir)

    print(f"🚀 Starting {symbol} {interval} data pipeline ({years} years) - {exchange.upper()}")
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
    print(f"📊 {exchange.upper()} PIPELINE EXECUTION SUMMARY")
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


# Backward compatibility function
async def run_bingx_klines_pipeline(
    symbol: str = "ETHUSDT",
    years: int = None,
    interval: str = "1m",
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    max_gap_minutes: int = 1,
    create_consolidated: bool = True
) -> Dict[str, Any]:
    """Run the complete pipeline for downloading klines data for any symbol from BingX.
    
    This is a backward compatibility function that calls the new exchange-agnostic function.
    """
    return await run_exchange_klines_pipeline(
        exchange="bingx",
        symbol=symbol,
        years=years,
        interval=interval,
        data_dir=data_dir,
        api_key=api_key,
        api_secret=api_secret,
        max_gap_minutes=max_gap_minutes,
        create_consolidated=create_consolidated
    )


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
        
        print(f"Starting {symbol} {years}-year 1m klines data download pipeline... (BingX)")

        # Run the complete pipeline
        results = await run_exchange_klines_pipeline(exchange="bingx", symbol=symbol, years=years)

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