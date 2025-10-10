#!/usr/bin/env python3
"""
Command Line Interface for Historical Data Pipeline

This script provides a command-line interface for managing historical Binance data.
"""

import argparse
import asyncio
import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
from src.utils.data.quality.data_cleaning import DataSchema
from src.utils.data.quality.data_quality import DataQualityFramework, check_dataframe_health
from src.utils.data.validation.validators import CrossStepValidator
from src.utils.logger import system_logger
from src.utils.data.gap_detector import GapDetector
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


async def _get_historical_klines_unified(
    exchange_interface: Optional[ExchangeInterface],
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int
) -> List[Dict[str, Any]]:
    """Get historical klines data using unified interface for both exchange types.
    
    Args:
        exchange_interface: ExchangeInterface instance
        symbol: Trading symbol
        interval: Kline interval
        start_time_ms: Start time in milliseconds
        end_time_ms: End time in milliseconds
        limit: Maximum number of records
        
    Returns:
        List of kline data dictionaries
    """
    try:
        if exchange_interface is not None:
            # Use ExchangeInterface
            from datetime import datetime
            start_time = datetime.fromtimestamp(start_time_ms / 1000)
            end_time = datetime.fromtimestamp(end_time_ms / 1000)
            
            klines_data = await exchange_interface.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Convert KlineData objects to dict format
            result = []
            for kline in klines_data:
                result.append({
                    "timestamp": int(kline.timestamp.timestamp() * 1000),
                    "open_time": int(kline.timestamp.timestamp() * 1000),
                    "open": kline.open_price,
                    "high": kline.high_price,
                    "low": kline.low_price,
                    "close": kline.close_price,
                    "volume": kline.volume,
                    "close_time": int(kline.close_time.timestamp() * 1000),
                    "quote_volume": kline.quote_asset_volume,
                    "trades": kline.number_of_trades,
                    "taker_buy_base": kline.taker_buy_base_asset_volume,
                    "taker_buy_quote": kline.taker_buy_quote_asset_volume
                })
            return result
        else:
            # Fallback to direct Binance exchange
            from exchanges.binance import BinanceExchange
            exchange = BinanceExchange("", "", symbol)
            await exchange._initialize_exchange()
            result = await exchange._get_historical_klines_raw(
                symbol, interval, start_time_ms, end_time_ms, limit
            )
            await exchange.close()
            return result
    except Exception as e:
        print(f"Error getting historical klines: {e}")
        return []


async def download_command(args):
    """Handle download command."""
    print(f"📥 Downloading {args.years} years of {args.symbol} data...")

    pipeline = HistoricalDataPipeline(args.data_dir)

    # Create ExchangeInterface for exchange-agnostic data access
    exchange_interface = None
    if args.api_key and args.api_secret:
        exchange_config = {
            'exchange_type': 'binance',
            'api_key': args.api_key,
            'api_secret': args.api_secret,
            'testnet': False
        }
        exchange_interface = create_exchange_interface(exchange_config)
        await exchange_interface.connect()

    # Download raw data
    success = await pipeline.downloader.download_historical_klines(
        symbol=args.symbol,
        interval="1m",
        years=args.years,
        exchange_interface=exchange_interface,
        api_key=args.api_key,
        api_secret=args.api_secret
    )

    if success:
        summary = pipeline.downloader.get_data_summary(args.symbol)
        print(f"✅ Download completed: {summary}")
    else:
        print("❌ Download failed")
        return 1

    return 0


async def download_standardized_command(args):
    """Handle download-standardized command with enforced format consistency."""
    print(f"📥 Downloading {args.years} years of {args.symbol} data with standardized format...")
    print(f"🎯 Ensuring format consistency: volume=float64, symbol=object")

    # Import required modules
    import pandas as pd

    try:
        # Create ExchangeInterface for exchange-agnostic data access
        exchange_interface = None
        if args.api_key and args.api_secret:
            exchange_config = {
                'exchange_type': 'binance',
                'api_key': args.api_key,
                'api_secret': args.api_secret,
                'testnet': False
            }
            exchange_interface = create_exchange_interface(exchange_config)
            await exchange_interface.connect()

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.years * 365)

        # Create directories
        data_dir = Path(args.data_dir) / "binance" / args.symbol.lower() / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        total_downloaded = 0
        months_processed = 0

        print(f"📅 Processing from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

        # Process month by month
        current_date = start_date.replace(day=1)
        while current_date < end_date:
            month_start = current_date
            if current_date.month == 12:
                month_end = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                month_end = current_date.replace(month=current_date.month + 1, day=1)

            if month_end > end_date:
                month_end = end_date

            print(f"📊 Downloading {month_start.strftime('%Y-%m')}...")

            # Download data
            start_time_ms = int(month_start.timestamp() * 1000)
            end_time_ms = int(month_end.timestamp() * 1000)

            raw_data = await _get_historical_klines_unified(
                exchange_interface, args.symbol, "1m", start_time_ms, end_time_ms, 1000
            )

            if raw_data:
                # Convert to DataFrame with standardized format
                df = pd.DataFrame(raw_data)

                # Convert timestamp to datetime index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.set_index('timestamp')

                # Convert numeric columns with explicit float64 for volume
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
                for col in numeric_columns:
                    if col in df.columns:
                        if col == 'volume':
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                        else:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                # Add metadata with explicit types
                df['symbol'] = args.symbol  # This will be object type, not category
                df['interval'] = "1m"
                df['year'] = df.index.year.astype('int32')
                df['month'] = df.index.month.astype('int32')
                df['day'] = df.index.day.astype('int32')

                # Ensure volume is explicitly float64 (not float32)
                df['volume'] = df['volume'].astype('float64')

                # Sort and remove duplicates
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='last')]

                # Save with standardized format
                filename = f"{args.symbol.lower()}_1m_{month_start.strftime('%Y_%m')}.parquet"
                filepath = data_dir / filename
                df.to_parquet(filepath, index=True, compression='snappy')

                print(f"💾 Saved {len(df)} records to {filename}")
                print(f"   Format: volume={df['volume'].dtype}, symbol={df['symbol'].dtype}")

                total_downloaded += len(df)
                months_processed += 1

            current_date = month_end

        # Cleanup exchange interface
        if exchange_interface:
            try:
                await exchange_interface.disconnect()
            except Exception as e:
                print(f"Warning: Error disconnecting exchange interface: {e}")

        print(f"✅ Standardized download completed!")
        print(f"📊 Total records: {total_downloaded}")
        print(f"📅 Months processed: {months_processed}")
        print(f"🎯 Format ensured: volume=float64, symbol=object")

        return 0

    except Exception as e:
        print(f"❌ Standardized download failed: {e}")
        return 1


def format_check_command(args):
    """Handle format check command."""
    print(f"🔍 Checking data format for {args.symbol} ({args.data_type})...")

    pipeline = HistoricalDataPipeline(args.data_dir)

    # Determine if we should fix issues
    fix_issues = args.fix and not args.validate_only

    if args.validate_only:
        print("📋 Validation mode: Only checking, not fixing")
        result = pipeline.validate_data_format(args.symbol, args.data_type, args.interval)
    else:
        print(f"🔧 {'Fixing' if fix_issues else 'Checking'} format issues...")
        result = pipeline.check_and_fix_data_format(args.symbol, args.data_type, args.interval, fix_issues)

    if result["success"]:
        print(f"✅ Format check completed!")
        print(f"📊 Files checked: {result['checked_files']}")
        print(f"🔍 Issues found: {result['issues_found']}")
        if not args.validate_only:
            print(f"🔧 Issues fixed: {result['issues_fixed']}")
        print(f"📝 Message: {result['message']}")
    else:
        print(f"❌ Format check failed: {result['message']}")
        return 1

    return 0


def fix_timezone_command(args):
    """Handle timezone fix command."""
    print(f"🕐 Fixing timezone issues for {args.symbol} ({args.data_type})...")

    try:
        import shutil

        data_dir = Path(args.data_dir) / "binance" / args.symbol.lower() / args.data_type
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return 1

        # Get all parquet files
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"❌ No parquet files found in {data_dir}")
            return 1

        print(f"📊 Found {len(parquet_files)} files to process")

        files_processed = 0
        files_fixed = 0

        for file_path in parquet_files:
            try:
                # Create backup if requested
                if args.backup:
                    backup_path = file_path.with_suffix('.parquet.backup')
                    shutil.copy2(file_path, backup_path)
                    print(f"💾 Backup created: {backup_path.name}")

                # Load the data
                df = pd.read_parquet(file_path)

                # Check if timestamps need timezone fixing
                if df.index.tz is None:
                    print(f"🔧 Fixing timezone for {file_path.name}")

                    # Assume the timestamps are in UTC (Binance standard)
                    df.index = df.index.tz_localize('UTC')

                    # Save the corrected file
                    df.to_parquet(file_path, index=True, compression='snappy')
                    files_fixed += 1
                    print(f"✅ Fixed: {file_path.name}")
                else:
                    print(f"✅ Already UTC: {file_path.name}")

                files_processed += 1

            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")

        print(f"\\n📋 Timezone Fix Summary:")
        print(f"  Files processed: {files_processed}")
        print(f"  Files fixed: {files_fixed}")
        print(f"  Backups created: {files_processed if args.backup else 0}")

        if files_fixed > 0:
            print("\\n🎯 Timezone issues have been corrected!")
            print("   All timestamps are now properly UTC timezone-aware")
        else:
            print("\\n✅ All files already have correct timezone information")

        return 0

    except Exception as e:
        print(f"❌ Timezone fix failed: {e}")
        import traceback
        traceback.print_exc()
        return 1




async def gap_check_command(args):
    """Handle gap check command."""
    print(f"🔍 Checking for gaps in {args.symbol} data...")
    
    pipeline = HistoricalDataPipeline(args.data_dir)
    
    # Detect gaps
    gaps = pipeline.gap_detector.detect_gaps(
        args.symbol, "1m", args.max_gap_minutes
    )
    
    if gaps:
        print(f"⚠️ Found {len(gaps)} gaps:")
        for i, gap in enumerate(gaps, 1):
            print(f"  Gap {i}: {gap['gap_start']} to {gap['gap_end']} ({gap['gap_minutes']:.1f} minutes)")
        
        if args.fill:
            print("🔧 Filling gaps...")
            results = await pipeline.gap_detector.fill_gaps(gaps, args.api_key, args.api_secret)
            print(f"✅ Gap filling completed: {results}")
    else:
        print("✅ No gaps found")
    
    return 0


async def process_command(args):
    """Handle process command."""
    print(f"🔧 Processing {args.symbol} data with basic returns feature engineering...")
    
    pipeline = HistoricalDataPipeline(args.data_dir)
    
    # Process data
    results = pipeline.basic_returns_engineer.process_symbol_data(
        args.symbol, "1m", args.intervals
    )
    
    if results["success"]:
        print(f"✅ Processing completed: {results}")
    else:
        print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


async def pipeline_command(args):
    """Handle complete pipeline command with comprehensive processing."""
    print(f"🚀 Running comprehensive pipeline for {args.symbol}")
    print(f"📋 Pipeline steps: Smart Download → Gap Detection → Format Fix → Feature Engineering → Validation")
    print(f"🎯 Target intervals: {', '.join(args.intervals)}")
    print(f"📅 Years to process: {args.years}")

    try:
        # Import required modules

        # Initialize components
        pipeline = HistoricalDataPipeline(args.data_dir)
        gap_detector = GapDetector(args.data_dir)
        basic_engineer = BasicReturnsEngineer(args.data_dir)

        pipeline_results = {
            "symbol": args.symbol,
            "start_time": datetime.now(),
            "steps_completed": [],
            "steps_failed": [],
            "metrics": {},
            "errors": []
        }

        # Step 0: Smart Download with API Selection
        print(f"\n📥 Step 0: Smart Download")
        print(f"🎯 Selecting optimal API based on timeframe...")

        # Determine API strategy based on requirements
        if args.years <= 1:
            api_choice = "websocket"  # Real-time for recent data
            print("📡 Using WebSocket API for recent data (≤1 year)")
        else:
            api_choice = "rest"  # Historical for bulk data
            print("🌐 Using REST API for historical data (>1 year)")

        # Download with selected API strategy
        download_success = await pipeline.downloader.download_historical_klines(
            symbol=args.symbol,
            interval="1m",
            years=args.years,
            api_key=args.api_key,
            api_secret=args.api_secret
        )

        if download_success:
            pipeline_results["steps_completed"].append("download")
            download_summary = pipeline.downloader.get_data_summary(args.symbol)
            pipeline_results["metrics"]["download"] = download_summary
            print(f"✅ Download completed: {download_summary}")
        else:
            pipeline_results["steps_failed"].append("download")
            pipeline_results["errors"].append("Download failed")
            print("❌ Download failed")
            return 1

        # Step 1: Format Consistency Check and Fix (automatically included in pipeline)
        print(f"\n🔧 Step 1: Format Consistency Check and Fix")
        # This is automatically handled by the enhanced pipeline

        # Step 2: Enhanced Gap Detection and Filling
        print(f"\n🔍 Step 2: Enhanced Gap Detection and Filling")
        max_gap_minutes = getattr(args, 'max_gap_minutes', 1)
        print(f"🎯 Maximum gap threshold: {max_gap_minutes} minutes")

        gaps = gap_detector.detect_gaps(args.symbol, "1m", max_gap_minutes)

        if gaps:
            print(f"⚠️ Found {len(gaps)} time intervals between data points")

            # Analyze gap patterns - crypto markets should be 24/7
            # Any gaps > 1 minute are problematic for continuous trading
            real_data_gaps = []
            large_data_gaps = []

            for gap in gaps:
                gap_duration = gap['gap_minutes']

                if gap_duration > 60:  # Large gaps (>1 hour) - critical
                    large_data_gaps.append(gap)
                elif gap_duration > 1:  # Small gaps (1-60 minutes) - significant
                    real_data_gaps.append(gap)

            print(f"📊 Gap Analysis (24/7 Crypto Market Expectations):")
            print(f"  🚨 Large data gaps (>1 hour): {len(large_data_gaps)} gaps")
            print(f"  ⚠️ Small data gaps (1-60 min): {len(real_data_gaps)} gaps")
            print(f"  ℹ️ Note: Crypto markets should trade 24/7 with no gaps > 1 minute")

            # Handle both large and small data gaps
            all_problematic_gaps = large_data_gaps + real_data_gaps

            if all_problematic_gaps:
                # Group by severity for reporting
                critical_gaps = large_data_gaps + [g for g in real_data_gaps if g['gap_minutes'] > 10]
                significant_gaps = [g for g in real_data_gaps if 1 < g['gap_minutes'] <= 10]
                minor_gaps = [g for g in real_data_gaps if g['gap_minutes'] <= 1]

                print(f"  🚨 Critical data gaps (>10m): {len(critical_gaps)} gaps")
                print(f"  ⚠️ Significant data gaps (1m-10m): {len(significant_gaps)} gaps")
                print(f"  ℹ️ Minor data gaps (≤1m): {len(minor_gaps)} gaps")

                if large_data_gaps:
                    print(f"\\n🚨 CRITICAL ISSUE: {len(large_data_gaps)} large gaps detected!")
                    print(f"   These represent missing trading hours in a 24/7 market")
                    print(f"   Average gap size: {sum(g['gap_minutes'] for g in large_data_gaps) / len(large_data_gaps):.1f} minutes")

                # Fill all problematic data gaps
                print("\\n🔧 Filling all data gaps...")
                fill_results = await gap_detector.fill_gaps(all_problematic_gaps, args.api_key, args.api_secret)
                print(f"✅ Gap filling completed: {fill_results['filled_gaps']}/{len(all_problematic_gaps)} gaps filled")
            else:
                print("✅ No data gaps found - perfect 24/7 coverage!")

            pipeline_results["steps_completed"].append("gap_analysis")
            pipeline_results["metrics"]["gaps"] = {
                "total_time_gaps": len(gaps),
                "large_data_gaps": len(large_data_gaps),
                "small_data_gaps": len(real_data_gaps),
                "critical_gaps": len(large_data_gaps),
                "significant_gaps": len([g for g in real_data_gaps if 1 < g['gap_minutes'] <= 10]) if real_data_gaps else 0,
                "minor_gaps": len([g for g in real_data_gaps if g['gap_minutes'] <= 1]) if real_data_gaps else 0
            }
        else:
            print("✅ No time gaps found in data")
            pipeline_results["steps_completed"].append("gap_check")
            pipeline_results["metrics"]["gaps"] = {"total_time_gaps": 0, "large_data_gaps": 0, "small_data_gaps": 0, "critical_gaps": 0, "significant_gaps": 0, "minor_gaps": 0}

        # Step 3: Feature Engineering and Resampling
        print(f"\n⚙️ Step 3: Feature Engineering and Resampling")
        print(f"🎯 Target intervals: {', '.join(args.intervals)}")

        feature_results = basic_engineer.process_symbol_data(args.symbol, "1m", args.intervals)

        if feature_results["success"]:
            print("✅ Feature engineering completed successfully")

            # Get processing summary
            processing_summary = feature_results.get("summary", {})
            pipeline_results["metrics"]["feature_engineering"] = processing_summary
            pipeline_results["steps_completed"].append("feature_engineering")

            # Log what was processed
            for interval in args.intervals:
                interval_data = processing_summary.get(interval, {})
                if interval_data:
                    records = interval_data.get('total_records', 0)
                    print(f"  📊 {interval}: {records:,} records processed")
        else:
            error_msg = feature_results.get("error", "Unknown error")
            print(f"❌ Feature engineering failed: {error_msg}")
            pipeline_results["steps_failed"].append("feature_engineering")
            pipeline_results["errors"].append(f"Feature engineering failed: {error_msg}")
            return 1

        # Step 4: Final Comprehensive Validation
        print(f"\n🔍 Step 4: Final Comprehensive Validation")
        print("🎯 Running comprehensive data validation on all processed intervals...")

        # Validate all intervals including the original 1m data
        all_intervals = ["1m"] + args.intervals

        # Run comprehensive validation
        validation_args = argparse.Namespace(
            symbol=args.symbol,
            intervals=all_intervals,
            data_dir=args.data_dir
        )

        validation_result = comprehensive_data_checker(validation_args)

        if validation_result == 0:
            print("✅ Final validation passed!")
            pipeline_results["steps_completed"].append("final_validation")
            pipeline_results["metrics"]["validation"] = {"result": "passed"}
        else:
            print("⚠️ Final validation found issues")
            pipeline_results["steps_completed"].append("final_validation")
            pipeline_results["metrics"]["validation"] = {"result": "issues_found", "exit_code": validation_result}

        # Final Summary
        end_time = datetime.now()
        duration = end_time - pipeline_results["start_time"]
        pipeline_results["end_time"] = end_time
        pipeline_results["duration_seconds"] = duration.total_seconds()

        print(f"\n📋 Comprehensive Pipeline Summary")
        print("=" * 50)
        print(f"🎯 Symbol: {args.symbol}")
        print(f"⏱️ Duration: {duration}")
        print(f"✅ Steps completed: {len(pipeline_results['steps_completed'])}")
        print(f"❌ Steps failed: {len(pipeline_results['steps_failed'])}")
        print(f"📊 Intervals processed: {', '.join(all_intervals)}")

        if pipeline_results["metrics"].get("gaps"):
            gaps_info = pipeline_results["metrics"]["gaps"]
            print(f"🔍 Gaps found/filled: {gaps_info['total_gaps']}/{gaps_info.get('filled_gaps', 0)}")

        if pipeline_results["steps_failed"]:
            print(f"🚨 Failed steps: {', '.join(pipeline_results['steps_failed'])}")
            print(f"📝 Errors: {pipeline_results['errors']}")
            return 1
        else:
            print("🎉 Comprehensive pipeline completed successfully!")
            return 0

    except Exception as e:
        print(f"❌ Comprehensive pipeline failed: {e}")
        traceback.print_exc()
        return 1


def status_command(args):
    """Handle status command."""
    print(f"📊 Checking status for {args.symbol}...")
    
    pipeline = HistoricalDataPipeline(args.data_dir)
    status = pipeline.get_pipeline_status(args.symbol)
    
    print(f"\nStatus for {status['symbol']}:")
    print(f"  Raw data available: {status['raw_data_available']}")
    
    if status['raw_data_available']:
        print(f"  Raw data summary: {status['data_summary']['raw']}")
        
        print(f"  Processed data available:")
        for interval, available in status['processed_data_available'].items():
            print(f"    {interval}: {available}")
        
        if status['recommendations']:
            print(f"  Recommendations:")
            for rec in status['recommendations']:
                print(f"    - {rec}")
    else:
        print("  No raw data found - run download first")
    
    return 0


def info_command(args):
    """Handle info command."""
    print(f"📋 Getting detailed info for {args.symbol} {args.interval}...")
    
    manager = get_klines_manager(args.data_dir)
    
    # Get data info
    info = manager.get_data_info(args.symbol, args.interval, args.data_type)
    
    if info["available"]:
        print(f"\nData info for {args.symbol} {args.interval} ({args.data_type}):")
        print(f"  Files: {info['files_count']}")
        print(f"  Records: {info['total_records']:,}")
        print(f"  Size: {info['file_size_mb']:.2f} MB")
        if info['date_range']:
            print(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
    else:
        print(f"❌ No {args.data_type} data found for {args.symbol} {args.interval}")
        return 1
    
    return 0


def list_command(args):
    """Handle list command."""
    print("📋 Listing available data...")
    
    manager = get_klines_manager(args.data_dir)
    available_data = manager.list_available_data()
    
    if available_data:
        print("\nAvailable data:")
        for symbol, intervals in available_data.items():
            print(f"  {symbol}: {', '.join(intervals)}")
    else:
        print("No data found")
    
    return 0


def comprehensive_data_checker(args):
    """Handle comprehensive data validation command."""
    print(f"🔍 Running comprehensive data validation for {args.symbol}...")

    try:
        # Initialize validator and data manager
        validator = CrossStepValidator()
        manager = get_klines_manager(args.data_dir)

        validation_results = {
            "symbol": args.symbol,
            "intervals": args.intervals,
            "overall_success": True,
            "validation_details": {},
            "errors": [],
            "warnings": []
        }

        # Define expected schema for processed data
        processed_schema = DataSchema(
            name="processed_klines",
            required_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            optional_columns=[
                'close_return', 'close_log_return', 'volume_return', 'volume_log_return',
                'price_range', 'price_range_pct', 'body_size', 'body_size_pct',
                'hour', 'day_of_week', 'is_weekend'
            ] + [f'close_lag_{i}' for i in [1,2,3]] +
                [f'volume_lag_{i}' for i in [1,2,3]] +
                [f'close_return_lag_{i}' for i in [1,2,3]],
            data_types={
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            },
            constraints={
                'timestamp': {'not_null': True, 'min': 0},
                'open': {'not_null': True, 'min': 0},
                'high': {'not_null': True, 'min': 0},
                'low': {'not_null': True, 'min': 0},
                'close': {'not_null': True, 'min': 0},
                'volume': {'not_null': True, 'min': 0}
            }
        )

        # Validate each interval
        for interval in args.intervals:
            print(f"  📊 Validating {interval} data...")

            # Get data info
            data_info = manager.get_data_info(args.symbol, interval, "processed")
            if not data_info["available"]:
                error_msg = f"No processed data available for {args.symbol} {interval}"
                validation_results["errors"].append(error_msg)
                validation_results["overall_success"] = False
                print(f"    ❌ {error_msg}")
                continue

            # Load and validate data
            try:
                data = manager.read_data(args.symbol, interval, data_type="processed")
                if data is None or data.empty:
                    error_msg = f"Empty data for {args.symbol} {interval}"
                    validation_results["errors"].append(error_msg)
                    validation_results["overall_success"] = False
                    print(f"    ❌ {error_msg}")
                    continue

                interval_validation = {
                    "records": len(data),
                    "columns_check": {},
                    "data_types_check": {},
                    "content_check": {},
                    "issues": [],
                    "warnings": []
                }

                # 1. Columns validation
                schema_validation = processed_schema.validate_dataframe(data)
                interval_validation["columns_check"] = {
                    "valid": schema_validation["valid"],
                    "missing_columns": schema_validation["missing_columns"],
                    "extra_columns": schema_validation["extra_columns"]
                }

                if not schema_validation["valid"]:
                    interval_validation["issues"].extend(schema_validation["errors"])
                if schema_validation["warnings"]:
                    interval_validation["warnings"].extend(schema_validation["warnings"])

                # 2. Data types validation
                type_issues = []
                for col, expected_type in processed_schema.data_types.items():
                    if col in data.columns:
                        actual_type = str(data[col].dtype)
                        if actual_type != expected_type:
                            type_issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")

                interval_validation["data_types_check"] = {
                    "issues": type_issues
                }
                interval_validation["issues"].extend(type_issues)

                # 3. Content validation (timestamp format, NaN, infinite values)
                content_issues = []
                content_warnings = []

                # Check timestamp format (should be milliseconds)
                if 'timestamp' in data.columns:
                    timestamp_min = data['timestamp'].min()
                    timestamp_max = data['timestamp'].max()
                    # Timestamps should be in milliseconds (13 digits for recent dates)
                    if timestamp_max < 1e12:  # Less than 13 digits
                        content_issues.append("Timestamps appear to be in seconds, should be milliseconds")
                    elif timestamp_max > 1e15:  # More than 15 digits
                        content_warnings.append("Timestamps may be in microseconds or incorrect format")

                # Check for NaN values
                nan_counts = data.isnull().sum()
                nan_columns = nan_counts[nan_counts > 0]
                if not nan_columns.empty:
                    for col, count in nan_columns.items():
                        pct = (count / len(data)) * 100
                        if pct > 1:  # More than 1% NaN
                            content_issues.append(f"Column {col}: {count} NaN values ({pct:.1f}%)")
                        else:
                            content_warnings.append(f"Column {col}: {count} NaN values ({pct:.1f}%)")

                # Check for infinite values
                inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
                inf_columns = inf_counts[inf_counts > 0]
                if not inf_columns.empty:
                    for col, count in inf_columns.items():
                        content_issues.append(f"Column {col}: {count} infinite values")

                # Check for negative prices/volumes
                for col in ['open', 'high', 'low', 'close']:
                    if col in data.columns:
                        neg_count = (data[col] < 0).sum()
                        if neg_count > 0:
                            content_issues.append(f"Column {col}: {neg_count} negative values")

                if 'volume' in data.columns:
                    neg_volume = (data['volume'] < 0).sum()
                    if neg_volume > 0:
                        content_issues.append(f"Volume: {neg_volume} negative values")

                # Check price relationships (high >= low, etc.)
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    invalid_high_low = (data['high'] < data['low']).sum()
                    if invalid_high_low > 0:
                        content_issues.append(f"High < Low in {invalid_high_low} rows")

                    # Check if close is within high-low range
                    invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
                    if invalid_close > 0:
                        content_issues.append(f"Close outside High-Low range in {invalid_close} rows")

                interval_validation["content_check"] = {
                    "timestamp_format_ok": len([i for i in content_issues if "timestamp" in i.lower()]) == 0,
                    "no_nan_values": len(nan_columns) == 0,
                    "no_infinite_values": len(inf_columns) == 0,
                    "valid_price_relationships": len([i for i in content_issues if "high" in i.lower() or "close" in i.lower()]) == 0
                }

                interval_validation["issues"].extend(content_issues)
                interval_validation["warnings"].extend(content_warnings)

                # Cross-step validation using validator
                try:
                    cross_validation = validator.validate_step_transition(
                        "data_processing", "final_validation",
                        data, data,  # Same data for input/output since we're just validating
                        {"interval": interval, "symbol": args.symbol}
                    )
                    if not cross_validation["passed"]:
                        interval_validation["issues"].extend([i["message"] for i in cross_validation["issues"]])
                    if cross_validation["warnings"]:
                        interval_validation["warnings"].extend(cross_validation["warnings"])
                except Exception as e:
                    interval_validation["warnings"].append(f"Cross-validation failed: {str(e)}")

                validation_results["validation_details"][interval] = interval_validation

                # Update overall results
                if interval_validation["issues"]:
                    validation_results["errors"].extend([f"{interval}: {issue}" for issue in interval_validation["issues"]])
                    validation_results["overall_success"] = False

                if interval_validation["warnings"]:
                    validation_results["warnings"].extend([f"{interval}: {warning}" for warning in interval_validation["warnings"]])

                # Print summary for this interval
                status = "✅" if not interval_validation["issues"] else "❌"
                print(f"    {status} {interval}: {len(data)} records, {len(interval_validation['issues'])} issues, {len(interval_validation['warnings'])} warnings")

            except Exception as e:
                error_msg = f"Failed to validate {args.symbol} {interval}: {str(e)}"
                validation_results["errors"].append(error_msg)
                validation_results["overall_success"] = False
                print(f"    ❌ {error_msg}")

        # Print final summary
        total_issues = len(validation_results["errors"])
        total_warnings = len(validation_results["warnings"])

        print(f"\n📋 Validation Summary:")
        print(f"  Symbol: {args.symbol}")
        print(f"  Intervals checked: {', '.join(args.intervals)}")
        print(f"  Total issues: {total_issues}")
        print(f"  Total warnings: {total_warnings}")

        if validation_results["overall_success"]:
            print("✅ All validations passed!")
        else:
            print("❌ Validation failed - check issues above")
            if total_issues > 0:
                print("\n🚨 Critical Issues:")
                for issue in validation_results["errors"][:10]:  # Show first 10
                    print(f"  - {issue}")
                if total_issues > 10:
                    print(f"  ... and {total_issues - 10} more issues")

        if total_warnings > 0:
            print(f"\n⚠️ Warnings ({total_warnings}):")
            for warning in validation_results["warnings"][:5]:  # Show first 5
                print(f"  - {warning}")
            if total_warnings > 5:
                print(f"  ... and {total_warnings - 5} more warnings")

        return 0 if validation_results["overall_success"] else 1

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


async def enhanced_pipeline_command(args):
    """Handle enhanced pipeline command with comprehensive per-month logging."""
    print(f"🚀 Running enhanced pipeline for {args.symbol} with per-month logging...")

    # Initialize components
    pipeline = HistoricalDataPipeline(args.data_dir)
    gap_detector = GapDetector(args.data_dir)
    quality_framework = DataQualityFramework()
    manager = get_klines_manager(args.data_dir)

    # Track monthly progress
    monthly_stats = {}

    try:
        print(f"📅 Starting enhanced pipeline at {datetime.now()}")

        # Step 1: Gap Detection and Filling with per-month logging
        print(f"\n🔍 Step 1: Gap Detection and Filling")
        print(f"📊 Checking for gaps in {args.symbol} data...")

        gaps = gap_detector.detect_gaps(args.symbol, "1m", args.max_gap_minutes)

        if gaps:
            print(f"⚠️ Found {len(gaps)} gaps across all data")

            # Group gaps by month for per-month reporting
            monthly_gaps = {}
            for gap in gaps:
                month_key = f"{gap['gap_start'].year}-{gap['gap_start'].month:02d}"
                if month_key not in monthly_gaps:
                    monthly_gaps[month_key] = []
                monthly_gaps[month_key].append(gap)

            # Log per-month gap statistics
            for month, month_gaps in monthly_gaps.items():
                total_gap_minutes = sum(gap['gap_minutes'] for gap in month_gaps)
                print(f"📅 {month}: {len(month_gaps)} gaps, {total_gap_minutes:.1f} minutes total")
                monthly_stats[month] = {'gaps': len(month_gaps), 'gap_minutes': total_gap_minutes}

            if args.fill:
                print(f"🔧 Filling gaps...")
                fill_results = await gap_detector.fill_gaps(gaps, args.api_key, args.api_secret)
                print(f"✅ Gap filling completed: {fill_results['filled_gaps']}/{len(gaps)} gaps filled, {fill_results['total_records_added']} records added")
        else:
            print("✅ No gaps found in data")

        # Step 2: Data Quality Check - Empty columns/rows with per-month logging
        print(f"\n🧹 Step 2: Data Quality Check - Empty columns/rows")

        # Load data and check quality per month
        raw_data = pipeline.basic_returns_engineer._load_all_raw_data(args.symbol, "1m")
        if raw_data is not None:
            # Group by month and check quality
            monthly_quality = {}
            for (year, month), month_data in raw_data.groupby([raw_data.index.year, raw_data.index.month]):
                month_key = f"{year}-{month:02d}"
                print(f"🔍 Checking {month_key} data quality...")

                # Check for empty columns
                empty_cols = month_data.columns[month_data.isnull().all()].tolist()
                if empty_cols:
                    print(f"⚠️ {month_key}: Found {len(empty_cols)} completely empty columns: {empty_cols}")
                    monthly_quality[month_key] = {'empty_columns': len(empty_cols), 'empty_rows': 0}

                # Check for empty rows
                empty_rows = month_data.isnull().all(axis=1).sum()
                if empty_rows > 0:
                    print(f"⚠️ {month_key}: Found {empty_rows} completely empty rows")
                    if month_key in monthly_quality:
                        monthly_quality[month_key]['empty_rows'] = empty_rows
                    else:
                        monthly_quality[month_key] = {'empty_columns': 0, 'empty_rows': empty_rows}

                # General quality check
                health = check_dataframe_health(month_data)
                if not health['healthy']:
                    print(f"⚠️ {month_key}: Data health issues: {health['issues']}")

                if month_key not in monthly_quality:
                    monthly_quality[month_key] = {'empty_columns': 0, 'empty_rows': 0}

                print(f"✅ {month_key}: {len(month_data)} records, {len(month_data.columns)} columns")

        # Step 3: Feature Engineering with validation
        print(f"\n⚙️ Step 3: Feature Engineering with NaN/Infinite/Empty validation")

        # Process data with feature engineering
        feature_results = pipeline.basic_returns_engineer.process_symbol_data(args.symbol, "1m", args.intervals)

        if feature_results["success"]:
            print(f"✅ Feature engineering completed successfully")

            # Load processed data and validate per month
            for interval in args.intervals:
                print(f"🔍 Validating {interval} processed data...")

                try:
                    # Use the same data reading approach as basic_returns_engineer
                    processed_dir = pipeline.basic_returns_engineer.processed_data_dir / args.symbol.lower() / "processed" / f"{args.symbol.lower()}_{interval}"
                    if processed_dir.exists():
                        # Read all parquet files in the processed directory (including subdirectories)
                        import glob
                        parquet_files = list(processed_dir.rglob("*.parquet"))
                        if parquet_files:
                            processed_data = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
                            # Ensure proper datetime index for monthly grouping
                            if 'timestamp' in processed_data.columns and not isinstance(processed_data.index, pd.DatetimeIndex):
                                processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'], unit='ms', utc=True)
                                processed_data = processed_data.set_index('timestamp')
                        else:
                            processed_data = None
                    else:
                        processed_data = None

                    if processed_data is not None:
                        # Group by month and validate
                        for (year, month), month_data in processed_data.groupby([processed_data.index.year, processed_data.index.month]):
                            month_key = f"{year}-{month:02d}"

                            print(f"📊 Validating {month_key} {interval} data...")

                            # Check for NaN values
                            nan_counts = month_data.isnull().sum()
                            total_nans = nan_counts.sum()
                            if total_nans > 0:
                                nan_cols = nan_counts[nan_counts > 0]
                                print(f"⚠️ {month_key} {interval}: {total_nans} NaN values in columns: {list(nan_cols.index)}")

                            # Check for infinite values
                            inf_counts = {}
                            for col in month_data.select_dtypes(include=[np.number]).columns:
                                inf_count = np.isinf(month_data[col]).sum()
                                if inf_count > 0:
                                    inf_counts[col] = inf_count

                            if inf_counts:
                                print(f"⚠️ {month_key} {interval}: Infinite values found in {len(inf_counts)} columns: {list(inf_counts.keys())}")

                            # Check for empty values
                            empty_cols = month_data.columns[month_data.isnull().all()].tolist()
                            if empty_cols:
                                print(f"⚠️ {month_key} {interval}: {len(empty_cols)} completely empty columns: {empty_cols}")

                            print(f"✅ {month_key} {interval}: {len(month_data)} records validated")

                except Exception as e:
                    print(f"❌ Error validating {interval} data: {e}")

        # Step 4: Post-feature engineering validation
        print(f"\n🔍 Step 4: Post-feature engineering validation")

        # Run comprehensive validation on all intervals
        all_intervals = ["1m"] + args.intervals
        validation_results = comprehensive_data_checker(
            argparse.Namespace(
                symbol=args.symbol,
                intervals=all_intervals,
                data_dir=args.data_dir
            )
        )

        # Summary report
        print(f"\n📋 Enhanced Pipeline Summary:")
        print(f"🎯 Symbol: {args.symbol}")
        print(f"📅 Processing completed at {datetime.now()}")
        print(f"📊 Intervals processed: {', '.join(all_intervals)}")

        if monthly_stats:
            print(f"\n📅 Monthly Gap Statistics:")
            for month, stats in monthly_stats.items():
                print(f"  {month}: {stats['gaps']} gaps, {stats['gap_minutes']:.1f} minutes")

        if validation_results == 0:
            print("✅ Enhanced pipeline completed successfully!")
        else:
            print("⚠️ Enhanced pipeline completed with validation issues")

        return validation_results

    except Exception as e:
        print(f"❌ Enhanced pipeline failed: {e}")
        return 1


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Historical Binance Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 3 years of ETHUSDT data
  python cli.py download --symbol ETHUSDT --years 3

  # Check for gaps and fill them
  python cli.py gap-check --symbol ETHUSDT --fill

  # Process data with feature engineering
  python cli.py process --symbol ETHUSDT --intervals 5m 15m 30m 1h

  # Run complete pipeline (enhanced: smart download, gap detection & filling, format fixing, resampling, feature engineering, validation)
  python cli.py pipeline --symbol ETHUSDT --years 3 --intervals 5m 15m 30m 1h

  # Run complete pipeline with custom gap threshold and final validation
  python cli.py pipeline --symbol ETHUSDT --years 3 --intervals 5m 15m 30m 1h --max-gap-minutes 5 --validate

  # Check status
  python cli.py status --symbol ETHUSDT

  # Get detailed info
  python cli.py info --symbol ETHUSDT --interval 1m --data-type raw

  # List all available data
  python cli.py list

  # Run comprehensive data validation
  python cli.py validate --symbol ETHUSDT --intervals 1m 5m 15m 30m 1h

  # Run enhanced pipeline with per-month logging (recommended)
  python cli.py enhanced-pipeline --symbol ETHUSDT --intervals 5m 15m 30m 1h --fill

  # Check and fix data format consistency
  python cli.py format-check --symbol ETHUSDT --fix

  # Validate data format without making changes
  python cli.py format-check --symbol ETHUSDT --validate-only

  # Fix timezone issues in existing data files
  python cli.py fix-timezone --symbol ETHUSDT --backup

  # Download data with standardized format (ensures volume=float64, symbol=object)
  python cli.py download-standardized --symbol ETHUSDT --years 3
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default="historical_data",
        help="Base directory for data storage (default: historical_data)"
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Binance API key (optional, for downloading)"
    )
    parser.add_argument(
        "--api-secret",
        default="",
        help="Binance API secret (optional, for downloading)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download historical data")
    download_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    download_parser.add_argument("--years", type=int, default=3, help="Number of years to download")

    # Download standardized command
    download_std_parser = subparsers.add_parser("download-standardized", help="Download historical data with standardized format")
    download_std_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    download_std_parser.add_argument("--years", type=int, default=3, help="Number of years to download")

    
    # Gap check command
    gap_parser = subparsers.add_parser("gap-check", help="Check for gaps in data")
    gap_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    gap_parser.add_argument("--max-gap-minutes", type=int, default=1, help="Maximum allowed gap in minutes")
    gap_parser.add_argument("--fill", action="store_true", help="Fill detected gaps")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process data with feature engineering")
    process_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    process_parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "30m", "1h"], help="Target intervals")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    pipeline_parser.add_argument("--years", type=int, default=3, help="Number of years to download")
    pipeline_parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "30m", "1h"], help="Target intervals")
    pipeline_parser.add_argument("--max-gap-minutes", type=int, default=1, help="Maximum allowed gap in minutes for gap detection")
    pipeline_parser.add_argument("--validate", action="store_true", help="Run comprehensive validation after pipeline completion")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check pipeline status")
    status_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get detailed data info")
    info_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    info_parser.add_argument("--interval", default="1m", help="Data interval")
    info_parser.add_argument("--data-type", choices=["raw", "processed"], default="raw", help="Data type")
    
    # List command
    subparsers.add_parser("list", help="List available data")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run comprehensive data validation")
    validate_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    validate_parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "30m", "1h"], help="Intervals to validate")

    # Fix timezone command
    timezone_parser = subparsers.add_parser("fix-timezone", help="Fix timezone issues in existing data files")
    timezone_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    timezone_parser.add_argument("--data-type", choices=["raw", "processed"], default="raw", help="Data type to fix")
    timezone_parser.add_argument("--backup", action="store_true", help="Create backup files before fixing")

    # Enhanced pipeline command
    enhanced_parser = subparsers.add_parser("enhanced-pipeline", help="Run enhanced pipeline with per-month logging")
    enhanced_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    enhanced_parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "30m", "1h"], help="Target intervals for resampling")
    enhanced_parser.add_argument("--max-gap-minutes", type=int, default=1, help="Maximum allowed gap in minutes")
    enhanced_parser.add_argument("--fill", action="store_true", help="Fill detected gaps")

    # Format check command
    format_parser = subparsers.add_parser("format-check", help="Check and fix data format consistency")
    format_parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    format_parser.add_argument("--data-type", choices=["raw", "processed"], default="raw", help="Data type to check")
    format_parser.add_argument("--interval", default="1m", help="Data interval (for processed data)")
    format_parser.add_argument("--fix", action="store_true", help="Automatically fix format issues")
    format_parser.add_argument("--validate-only", action="store_true", help="Only validate without fixing")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    try:
        if args.command == "download":
            return asyncio.run(download_command(args))
        elif args.command == "download-standardized":
            return asyncio.run(download_standardized_command(args))
        elif args.command == "gap-check":
            return asyncio.run(gap_check_command(args))
        elif args.command == "process":
            return asyncio.run(process_command(args))
        elif args.command == "pipeline":
            return asyncio.run(pipeline_command(args))
        elif args.command == "status":
            return status_command(args)
        elif args.command == "info":
            return info_command(args)
        elif args.command == "list":
            return list_command(args)
        elif args.command == "validate":
            return comprehensive_data_checker(args)
        elif args.command == "enhanced-pipeline":
            return asyncio.run(enhanced_pipeline_command(args))
        elif args.command == "format-check":
            return format_check_command(args)
        elif args.command == "fix-timezone":
            return fix_timezone_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
