#!/usr/bin/env python3
"""
Test script for optimized data downloader

This script demonstrates the optimized data downloading capabilities for MEXC and GATEIO.
It shows how the system:
1. Only downloads data that isn't already downloaded
2. Saves CSV files incrementally as data is received
3. Uses parallel processing for better performance

Usage:
    python scripts/test_optimized_download.py --symbol ETHUSDT --exchange MEXC
    python scripts/test_optimized_download.py --symbol ETHUSDT --exchange GATEIO
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.ares_data_downloader_optimized import OptimizedDataDownloader, DownloadConfig
from src.utils.logger import setup_logging


async def test_optimized_download(symbol: str, exchange: str, interval: str = "1m", lookback_years: int = 1):
    """Test the optimized data downloader."""
    print(f"🧪 Testing optimized data downloader")
    print(f"📊 Symbol: {symbol}")
    print(f"🏦 Exchange: {exchange}")
    print(f"⏱️ Interval: {interval}")
    print(f"📅 Lookback: {lookback_years} years")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Create configuration
    config = DownloadConfig(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        lookback_years=lookback_years,
        max_concurrent_downloads=3,  # Conservative for testing
        max_concurrent_requests=5
    )
    
    # Create downloader
    downloader = OptimizedDataDownloader(config)
    
    try:
        # Run optimized download
        success = await downloader.run_optimized_download()
        
        if success:
            print("✅ Optimized download test completed successfully!")
            print(f"📊 Statistics:")
            print(f"   📈 Klines downloaded: {downloader.stats['klines_downloaded']}")
            print(f"   📈 Aggtrades downloaded: {downloader.stats['aggtrades_downloaded']}")
            print(f"   📈 Futures downloaded: {downloader.stats['futures_downloaded']}")
            print(f"   ⏱️ Total time: {downloader.stats['total_time']:.2f} seconds")
            print(f"   ❌ Errors: {downloader.stats['errors']}")
            return True
        else:
            print("❌ Optimized download test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during optimized download test: {e}")
        return False


def main():
    """Main function for testing optimized downloader."""
    parser = argparse.ArgumentParser(
        description="Test optimized data downloader for Ares trading bot"
    )
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., ETHUSDT)")
    parser.add_argument("--exchange", type=str, required=True, choices=["MEXC", "GATEIO"], help="Exchange name")
    parser.add_argument("--interval", type=str, default="1m", help="K-line interval (default: 1m)")
    parser.add_argument("--lookback-years", type=int, default=1, help="Years of data to download (default: 1)")
    
    args = parser.parse_args()
    
    # Run test
    success = asyncio.run(test_optimized_download(
        symbol=args.symbol,
        exchange=args.exchange,
        interval=args.interval,
        lookback_years=args.lookback_years
    ))
    
    if success:
        print("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 