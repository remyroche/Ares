"""
Complete Historical Data Pipeline

This module provides a complete pipeline for downloading, processing, and managing
historical klines data from any supported exchange with gap detection, feature engineering, and resampling.
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.data.historical_data_downloader import HistoricalDataDownloader
from src.utils.data.gap_detector import GapDetector
from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
from src.utils.data.optimized_parquet_storage import OptimizedParquetStorage
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.data.quality.comprehensive_duplicate_analyzer import ComprehensiveDuplicateAnalyzer
from src.trading.execution.exchange_interface import ExchangeInterface

class HistoricalDataPipeline:
    """Complete pipeline for historical data management."""

    def __init__(self, data_dir: str = "historical_data", exchange: str = "binance"):
        """Initialize the historical data pipeline.

        Args:
            data_dir: Base directory for data storage
            exchange: Exchange name for data organization
        """
        self.data_dir = Path(data_dir)
        self.exchange = exchange.lower()
        self.logger = system_logger.getChild("HistoricalDataPipeline")

        # Initialize components
        self.downloader = HistoricalDataDownloader(data_dir, exchange)
        self.gap_detector = GapDetector(data_dir, exchange)
        self.basic_returns_engineer = BasicReturnsEngineer(data_dir)
        self.optimized_storage = OptimizedParquetStorage(data_dir)
        self.klines_manager = KlinesParquetManager(data_dir, exchange)
        self.duplicate_analyzer = ComprehensiveDuplicateAnalyzer(self.logger)

        # Columns to remove - keeping taker columns as requested
        self.columns_to_remove = []

    async def run_complete_pipeline(
        self,
        symbol: str = "ETHUSDT",
        years: int = 3,
        exchange_interface: Optional[ExchangeInterface] = None,
        api_key: str = "",
        api_secret: str = "",
        target_intervals: List[str] = None,
        max_gap_minutes: int = 1
    ) -> Dict[str, Any]:
        """Run the complete historical data pipeline.

        Args:
            symbol: Trading symbol
            years: Number of years to download
            exchange_interface: ExchangeInterface instance (preferred over api_key/api_secret)
            api_key: Exchange API key (fallback if exchange_interface not provided)
            api_secret: Exchange API secret (fallback if exchange_interface not provided)
            target_intervals: List of target intervals for resampling
            max_gap_minutes: Maximum allowed gap in minutes

        Returns:
            Dictionary with pipeline results
        """
        if target_intervals is None:
            target_intervals = ["5m", "15m", "30m", "1h"]

        try:
            self.logger.info(f"🚀 Starting complete pipeline for {symbol}")

            results = {
                "symbol": symbol,
                "years": years,
                "target_intervals": target_intervals,
                "steps_completed": [],
                "errors": [],
                "warnings": [],
                "summary": {}
            }

            # Step 1: Download historical data
            self.logger.info("📥 Step 1: Downloading historical data")
            download_success = await self.downloader.download_historical_klines(
                symbol=symbol,
                interval="1m",
                years=years,
                exchange_interface=exchange_interface,
                api_key=api_key,
                api_secret=api_secret
            )

            if download_success:
                results["steps_completed"].append("download")
                download_summary = self.downloader.get_data_summary(symbol)
                results["summary"]["download"] = download_summary
                self.logger.info(f"✅ Download completed: {download_summary}")

                # Step 1.5: Check and fix data format consistency
                self.logger.info("🔍 Step 1.5: Checking and fixing data format consistency")
                format_check_result = self.check_and_fix_data_format(symbol, "raw", "1m", fix_issues=True)
                results["steps_completed"].append("format_check")
                results["summary"]["format_check"] = format_check_result

                if format_check_result["issues_found"] > 0:
                    self.logger.info(f"🔧 Fixed {format_check_result['issues_fixed']} format issues in {format_check_result['checked_files']} files")
                else:
                    self.logger.info("✅ Data format is already consistent")

            else:
                results["errors"].append("Download failed")
                self.logger.error("❌ Download failed")
                return results

            # Step 2: Detect and fill gaps
            self.logger.info("🔍 Step 2: Detecting and filling gaps")
            gaps = self.gap_detector.detect_gaps(symbol, "1m", max_gap_minutes)
            self.gap_detector.log_gaps(gaps)

            if gaps:
                gap_results = await self.gap_detector.fill_gaps(gaps, exchange_interface, api_key, api_secret)
                results["steps_completed"].append("gap_filling")
                results["summary"]["gap_filling"] = gap_results
                self.logger.info(f"✅ Gap filling completed: {gap_results}")

            else:
                results["steps_completed"].append("gap_detection")
                results["summary"]["gap_detection"] = {"gaps_detected": 0}
                self.logger.info("✅ No gaps detected")

            # Step 2.7: Handle duplicates (warn on false, remove true)
            self.logger.info("🔍 Step 2.7: Analyzing and handling duplicate timestamps")
            duplicate_results = self.handle_duplicates(symbol, "1m")
            results["steps_completed"].append("duplicate_handling")
            results["summary"]["duplicate_handling"] = duplicate_results

            # Add warnings from duplicate analysis
            if duplicate_results.get("warnings"):
                results["warnings"] = results.get("warnings", []) + duplicate_results["warnings"]

            # Step 3: Basic returns feature engineering and resampling
            self.logger.info("🔧 Step 3: Basic returns feature engineering and resampling")
            processing_results = self.basic_returns_engineer.process_symbol_data(
                symbol, "1m", target_intervals
            )

            if processing_results["success"]:
                results["steps_completed"].append("feature_engineering")
                results["summary"]["feature_engineering"] = processing_results
                self.logger.info(f"✅ Feature engineering completed: {processing_results}")
            else:
                results["errors"].append(f"Basic returns feature engineering failed: {processing_results.get('error', 'Unknown error')}")
                self.logger.error(f"❌ Basic returns feature engineering failed: {processing_results.get('error', 'Unknown error')}")

            # Step 4: Verify data integrity
            self.logger.info("🔍 Step 4: Verifying data integrity")
            verification_results = self._verify_data_integrity(symbol, target_intervals)
            results["steps_completed"].append("verification")
            results["summary"]["verification"] = verification_results
            self.logger.info(f"✅ Verification completed: {verification_results}")

            # Final summary
            results["pipeline_success"] = len(results["errors"]) == 0
            results["completion_time"] = datetime.now().isoformat()

            self.logger.info(f"🎉 Pipeline completed: {len(results['steps_completed'])} steps, {len(results['errors'])} errors")
            return results

        except Exception as e:
            self.logger.exception(f"❌ Pipeline failed: {e}")
            results["errors"].append(str(e))
            results["pipeline_success"] = False
            return results

    def _verify_data_integrity(
        self,
        symbol: str,
        target_intervals: List[str]
    ) -> Dict[str, Any]:
        """Verify data integrity across all intervals.

        Args:
            symbol: Trading symbol
            target_intervals: List of target intervals

        Returns:
            Dictionary with verification results
        """
        try:
            verification_results = {
                "raw_data": {},
                "processed_data": {},
                "overall_success": True
            }

            # Check raw data
            raw_info = self.klines_manager.get_data_info(symbol, "1m", "raw")
            verification_results["raw_data"] = raw_info

            if not raw_info["available"]:
                verification_results["overall_success"] = False
                self.logger.error("❌ No raw data found")
                return verification_results

            # Check processed data for each interval
            all_intervals = ["1m"] + target_intervals
            for interval in all_intervals:
                processed_info = self.klines_manager.get_data_info(symbol, interval, "processed")
                verification_results["processed_data"][interval] = processed_info

                if not processed_info["available"]:
                    verification_results["overall_success"] = False
                    self.logger.error(f"❌ No processed data found for {interval}")

            # Check for data consistency
            if verification_results["overall_success"]:
                self.logger.info("✅ All data integrity checks passed")
            else:
                self.logger.warning("⚠️ Some data integrity checks failed")

            return verification_results

        except Exception as e:
            self.logger.exception(f"❌ Data integrity verification failed: {e}")
            return {
                "raw_data": {},
                "processed_data": {},
                "overall_success": False,
                "error": str(e)
            }

    def get_pipeline_status(self, symbol: str) -> Dict[str, Any]:
        """Get current pipeline status for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with pipeline status
        """
        try:
            status = {
                "symbol": symbol,
                "raw_data_available": False,
                "processed_data_available": {},
                "data_summary": {},
                "recommendations": []
            }

            # Check raw data
            raw_info = self.klines_manager.get_data_info(symbol, "1m", "raw")
            status["raw_data_available"] = raw_info["available"]
            status["data_summary"]["raw"] = raw_info

            if raw_info["available"]:
                # Check for gaps
                gaps = self.gap_detector.detect_gaps(symbol, "1m", max_gap_minutes=1)
                if gaps:
                    status["recommendations"].append(f"Found {len(gaps)} gaps in raw data - consider running gap filling")

                # Check processed data
                intervals = ["1m", "5m", "15m", "30m", "1h"]
                for interval in intervals:
                    processed_info = self.klines_manager.get_data_info(symbol, interval, "processed")
                    status["processed_data_available"][interval] = processed_info["available"]
                    status["data_summary"][f"processed_{interval}"] = processed_info

                # Generate recommendations
                if not any(status["processed_data_available"].values()):
                    status["recommendations"].append("No processed data found - consider running feature engineering")

                missing_intervals = [interval for interval, available in status["processed_data_available"].items() if not available]
                if missing_intervals:
                    status["recommendations"].append(f"Missing processed data for intervals: {missing_intervals}")
            else:
                status["recommendations"].append("No raw data found - consider running data download")

            return status

        except Exception as e:
            self.logger.exception(f"❌ Failed to get pipeline status: {e}")
            return {
                "symbol": symbol,
                "raw_data_available": False,
                "processed_data_available": {},
                "data_summary": {},
                "recommendations": [f"Error getting status: {e}"],
                "error": str(e)
            }

    def cleanup_old_data(
        self,
        symbol: str,
        keep_days: int = 30,
        data_type: str = "raw"
    ) -> Dict[str, Any]:
        """Clean up old data to save space.

        Args:
            symbol: Trading symbol
            keep_days: Number of days to keep
            data_type: 'raw' or 'processed'

        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            self.logger.info(f"🧹 Cleaning up {data_type} data older than {cutoff_date}")

            # Get data info
            info = self.klines_manager.get_data_info(symbol, "1m", data_type)

            if not info["available"]:
                return {"cleaned_files": 0, "freed_space_mb": 0, "message": "No data to clean"}

            # Delete old data
            success = self.klines_manager.delete_data(
                symbol, "1m", data_type, end_date=cutoff_date
            )

            if success:
                # Get new data info
                new_info = self.klines_manager.get_data_info(symbol, "1m", data_type)
                freed_space = info["file_size_mb"] - new_info["file_size_mb"]

                return {
                    "cleaned_files": info["files_count"] - new_info["files_count"],
                    "freed_space_mb": freed_space,
                    "message": f"Cleaned up {freed_space:.2f} MB of old data"
                }
            else:
                return {"cleaned_files": 0, "freed_space_mb": 0, "message": "Cleanup failed"}

        except Exception as e:
            self.logger.exception(f"❌ Cleanup failed: {e}")
            return {"cleaned_files": 0, "freed_space_mb": 0, "message": f"Cleanup failed: {e}"}

    def remove_unwanted_columns(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Remove unwanted columns from all data files.

        Note: As of the latest update, unwanted columns (taker_buy_base, taker_buy_quote, year)
        are filtered out during data download, so no column removal is needed.

        Args:
            symbol: Trading symbol
            interval: Data interval

        Returns:
            Dictionary with column removal results
        """
        self.logger.info(f"ℹ️ Column removal skipped - unwanted columns are filtered during download for {symbol} {interval}")

        return {
            "files_processed": 0,
            "columns_removed": 0,
            "columns_targeted": self.columns_to_remove,
            "success": True,
            "message": "No columns to remove - filtered during download"
        }

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

    def check_and_fix_data_format(
        self,
        symbol: str,
        data_type: str = "raw",
        interval: str = "1m",
        fix_issues: bool = True
    ) -> Dict[str, Any]:
        """Check and fix data format consistency issues.

        Args:
            symbol: Trading symbol
            data_type: 'raw' or 'processed'
            interval: Data interval
            fix_issues: Whether to automatically fix format issues

        Returns:
            Dictionary with format check results
        """
        try:
            self.logger.info(f"🔍 Checking data format consistency for {symbol} {interval} ({data_type})")

            # Get list of files for this symbol/interval
            if data_type == "raw":
                symbol_dir = self.data_dir / self.exchange / symbol.lower() / "raw"
            else:
                symbol_dir = self.data_dir / self.exchange / symbol.lower() / "processed" / f"{symbol.lower()}_{interval}"

            if not symbol_dir.exists():
                return {"checked_files": 0, "issues_found": 0, "issues_fixed": 0, "message": "No data directory found"}

            # Get all parquet files - handle both direct files and partitioned directories
            parquet_files = []
            if data_type == "raw":
                # For raw data, files are directly in the directory
                parquet_files = list(symbol_dir.glob("*.parquet"))
            else:
                # For processed data, files might be in partitioned subdirectories
                for root, dirs, files_in_dir in os.walk(symbol_dir):
                    for file in files_in_dir:
                        if file.endswith('.parquet'):
                            parquet_files.append(Path(root) / file)

            if not parquet_files:
                return {"checked_files": 0, "issues_found": 0, "issues_fixed": 0, "message": "No parquet files found"}

            self.logger.info(f"📊 Checking {len(parquet_files)} files for format consistency")

            issues_found = 0
            issues_fixed = 0
            files_checked = 0

            # Define expected format standards
            expected_format = {
                'volume': 'float64',
                'symbol': 'object',
                'open_time': 'int64',
                'close_time': 'int64',
                'open': 'float32',  # These can stay float32 for memory efficiency
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'quote_volume': 'float64',
                'trades': 'int16',
                'taker_buy_base': 'object',
                'taker_buy_quote': 'object',
                'interval': 'category',  # Can stay category for memory efficiency
                'year': 'int32',
                'month': 'int32',
                'day': 'int32'
            }

            for file_path in parquet_files:
                try:
                    files_checked += 1
                    self.logger.debug(f"🔍 Checking {file_path.name}")

                    # Read the file
                    df = pd.read_parquet(file_path)

                    if df.empty:
                        self.logger.warning(f"⚠️ Empty file: {file_path.name}")
                        continue

                    # Check for format issues
                    file_issues = []

                    for col, expected_dtype in expected_format.items():
                        if col in df.columns:
                            actual_dtype = str(df[col].dtype)

                            # Handle special cases
                            if col == 'symbol' and actual_dtype.startswith('category'):
                                # Category is acceptable for symbol column
                                continue
                            elif col in ['volume', 'quote_volume'] and actual_dtype == 'float32':
                                # Need to fix volume/quote_volume from float32 to float64
                                file_issues.append({
                                    'column': col,
                                    'expected': expected_dtype,
                                    'actual': actual_dtype,
                                    'fix': 'convert_to_float64'
                                })
                            elif col == 'symbol' and actual_dtype.startswith('category'):
                                # Category is fine for symbol, no issue
                                pass
                            elif actual_dtype != expected_dtype:
                                # General dtype mismatch
                                file_issues.append({
                                    'column': col,
                                    'expected': expected_dtype,
                                    'actual': actual_dtype,
                                    'fix': 'convert_dtype'
                                })

                    # If issues found and fix is enabled
                    if file_issues and fix_issues:
                        self.logger.info(f"🔧 Fixing {len(file_issues)} format issues in {file_path.name}")

                        for issue in file_issues:
                            col = issue['column']
                            if issue['fix'] == 'convert_to_float64':
                                df[col] = df[col].astype('float64')
                                self.logger.debug(f"  ✅ Converted {col} to float64")
                            elif issue['fix'] == 'convert_dtype':
                                try:
                                    df[col] = df[col].astype(issue['expected'])
                                    self.logger.debug(f"  ✅ Converted {col} to {issue['expected']}")
                                except Exception as e:
                                    self.logger.warning(f"  ⚠️ Could not convert {col}: {e}")

                        # Save the corrected file
                        df.to_parquet(file_path, index=True, compression='snappy')
                        issues_fixed += len(file_issues)
                        self.logger.info(f"💾 Saved corrected file: {file_path.name}")

                    issues_found += len(file_issues)

                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path.name}: {e}")

            result = {
                "checked_files": files_checked,
                "issues_found": issues_found,
                "issues_fixed": issues_fixed,
                "success": True,
                "message": f"Format check completed. Found {issues_found} issues, fixed {issues_fixed}"
            }

            if issues_found > 0:
                self.logger.info(f"🎯 Format consistency check: {issues_found} issues found, {issues_fixed} fixed")
            else:
                self.logger.info("✅ All files have consistent format")

            return result

        except Exception as e:
            self.logger.exception(f"❌ Format check failed: {e}")
            return {
                "checked_files": 0,
                "issues_found": 0,
                "issues_fixed": 0,
                "success": False,
                "message": f"Format check failed: {e}"
            }

    def validate_data_format(
        self,
        symbol: str,
        data_type: str = "raw",
        interval: str = "1m"
    ) -> Dict[str, Any]:
        """Validate data format without making changes.

        Args:
            symbol: Trading symbol
            data_type: 'raw' or 'processed'
            interval: Data interval

        Returns:
            Dictionary with validation results
        """
        return self.check_and_fix_data_format(symbol, data_type, interval, fix_issues=False)

# Convenience functions
async def run_ethusdt_pipeline(
    years: int = 3,
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    target_intervals: List[str] = None
) -> Dict[str, Any]:
    """Run the complete pipeline for ETHUSDT.

    Args:
        years: Number of years to download
        data_dir: Base directory for data storage
        api_key: Exchange API key
        api_secret: Exchange API secret
        target_intervals: List of target intervals for resampling

    Returns:
        Dictionary with pipeline results
    """
    if target_intervals is None:
        target_intervals = ["5m", "15m", "30m", "1h"]

    pipeline = HistoricalDataPipeline(data_dir)
    return await pipeline.run_complete_pipeline(
        symbol="ETHUSDT",
        years=years,
        api_key=api_key,
        api_secret=api_secret,
        target_intervals=target_intervals
    )

# Convenience functions for common use cases
async def run_ethusdt_3year_pipeline(
    data_dir: str = "historical_data",
    api_key: str = "",
    api_secret: str = "",
    target_intervals: List[str] = None,
    max_gap_minutes: int = 1
) -> Dict[str, Any]:
    """Run the complete pipeline for downloading 3 years of ETHUSDT 1m klines data.

    This function includes:
    - Downloads 3 years of ETHUSDT data using HistoricalDataPipeline (keeping all columns including taker data)
    - Detects gaps > 1m and re-downloads if needed
    - Analyzes duplicates (warns on false duplicates, removes true duplicates)
    - Runs feature engineering and resampling
    - Performs final data verification

    Args:
        data_dir: Base directory for data storage
        api_key: Exchange API key
        api_secret: Exchange API secret
        target_intervals: List of target intervals for resampling
        max_gap_minutes: Maximum allowed gap in minutes

    Returns:
        Dictionary with complete pipeline results
    """
    if target_intervals is None:
        target_intervals = ["5m", "15m", "30m", "1h"]

    pipeline = HistoricalDataPipeline(data_dir)

    print(f"🚀 Starting ETHUSDT {len(target_intervals)} interval pipeline (3 years)")
    print(f"📁 Data directory: {data_dir}")
    print(f"⏱️  Target intervals: {', '.join(target_intervals)}")
    print(f"🎯 Max gap threshold: {max_gap_minutes} minutes")
    print()

    results = await pipeline.run_complete_pipeline(
        symbol="ETHUSDT",
        years=3,
        api_key=api_key,
        api_secret=api_secret,
        target_intervals=target_intervals,
        max_gap_minutes=max_gap_minutes
    )

    # Print summary
    print("\n" + "="*80)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Symbol: {results['symbol']}")
    print(f"Years: {results['years']}")
    print(f"Target Intervals: {', '.join(results['target_intervals'])}")
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
    # Example usage - download 3 years of ETHUSDT data
    async def main():
        print("Starting ETHUSDT 3-year complete pipeline...")

        # Run the complete pipeline
        results = await run_ethusdt_3year_pipeline()

        print(f"\nPipeline completed with success: {results['pipeline_success']}")

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
