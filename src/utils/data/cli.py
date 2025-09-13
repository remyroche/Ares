#!/usr/bin/env python3
"""
Command Line Interface for Historical Data Pipeline

This script provides a command-line interface for managing historical Binance data.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data.historical_data_pipeline import HistoricalDataPipeline
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.data.basic_returns_engineer import BasicReturnsEngineer
from src.utils.logger import system_logger


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    else:
        import logging
        logging.basicConfig(level=logging.INFO)


async def download_command(args):
    """Handle download command."""
    print(f"📥 Downloading {args.years} years of {args.symbol} data...")
    
    pipeline = HistoricalDataPipeline(args.data_dir)
    
    # Download raw data
    success = await pipeline.downloader.download_historical_klines(
        symbol=args.symbol,
        interval="1m",
        years=args.years,
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
    """Handle complete pipeline command."""
    print(f"🚀 Running complete pipeline for {args.symbol}...")
    
    pipeline = HistoricalDataPipeline(args.data_dir)
    
    # Run complete pipeline
    results = await pipeline.run_complete_pipeline(
        symbol=args.symbol,
        years=args.years,
        api_key=args.api_key,
        api_secret=args.api_secret,
        target_intervals=args.intervals
    )
    
    if results["pipeline_success"]:
        print(f"✅ Pipeline completed successfully!")
        print(f"Steps completed: {results['steps_completed']}")
    else:
        print(f"❌ Pipeline failed with errors: {results['errors']}")
        return 1
    
    return 0


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

  # Run complete pipeline
  python cli.py pipeline --symbol ETHUSDT --years 3 --intervals 5m 15m 30m 1h

  # Check status
  python cli.py status --symbol ETHUSDT

  # Get detailed info
  python cli.py info --symbol ETHUSDT --interval 1m --data-type raw

  # List all available data
  python cli.py list
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