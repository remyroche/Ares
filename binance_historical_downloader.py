#!/usr/bin/env python3
"""
Binance Historical Data Downloader

This script downloads historical ETHUSDT 1-minute data directly from Binance's REST API
without requiring API keys. Uses the public klines endpoint.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import time
import json
import ssl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import system_logger

logger = system_logger.getChild('BinanceHistoricalDownloader')

class BinanceHistoricalDownloader:
    """Downloads historical ETHUSDT 1-minute data from Binance's public API."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "binance" / "ethusdt" / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Binance public API endpoints
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
    def analyze_existing_raw_data(self) -> Dict[str, Any]:
        """Analyze existing data in the raw directory."""
        if not self.raw_dir.exists():
            return {
                "total_files": 0,
                "date_ranges": [],
                "coverage_percentage": 0.0
            }
        
        # Find all parquet files in raw directory
        raw_files = list(self.raw_dir.glob("ethusdt_1m_*.parquet"))
        
        date_ranges = []
        for file in raw_files:
            try:
                # Extract date from filename (format: ethusdt_1m_YYYY_MM.parquet)
                filename = file.name
                if "ethusdt_1m_" in filename and filename.endswith(".parquet"):
                    # Extract year and month from filename
                    parts = filename.replace("ethusdt_1m_", "").replace(".parquet", "").split("_")
                    if len(parts) == 2:
                        year, month = int(parts[0]), int(parts[1])
                        # Calculate start and end of month
                        month_start = datetime(year, month, 1)
                        if month == 12:
                            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                        else:
                            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
                        
                        # Get file size for record estimation
                        file_size = file.stat().st_size
                        estimated_records = file_size // 100  # Rough estimate: 100 bytes per record
                        
                        date_ranges.append({
                            'file': file.name,
                            'start': month_start,
                            'end': month_end,
                            'records': estimated_records,
                            'size_mb': file_size / (1024 * 1024)
                        })
            except Exception as e:
                logger.warning(f"Could not analyze file {file}: {e}")
        
        # Sort by start date
        date_ranges.sort(key=lambda x: x['start'])
        
        return {
            "total_files": len(raw_files),
            "date_ranges": date_ranges,
            "analyzed_files": len(date_ranges)
        }
    
    def identify_missing_periods(self, target_start: datetime, target_end: datetime) -> List[Tuple[datetime, datetime]]:
        """Identify missing periods by analyzing existing raw data."""
        analysis = self.analyze_existing_raw_data()
        date_ranges = analysis['date_ranges']
        
        print(f"📊 Found {analysis['total_files']} existing raw data files")
        print(f"📈 Analyzed {analysis['analyzed_files']} files successfully")
        
        if not date_ranges:
            # No existing data, need to download everything
            return [(target_start, target_end)]
        
        # Show existing coverage
        print(f"\n📋 Existing data coverage:")
        for i, range_info in enumerate(date_ranges[:10], 1):  # Show first 10
            print(f"  {i}. {range_info['start'].strftime('%Y-%m')} ({range_info['records']:,} records, {range_info['size_mb']:.1f} MB)")
        
        if len(date_ranges) > 10:
            print(f"  ... and {len(date_ranges) - 10} more months")
        
        missing_periods = []
        current_time = target_start
        
        for range_info in date_ranges:
            range_start = range_info['start']
            range_end = range_info['end']
            
            # Check if there's a gap before this range
            if current_time < range_start:
                gap_end = min(range_start, target_end)
                if current_time < gap_end:
                    missing_periods.append((current_time, gap_end))
            
            # Move current_time to after this range
            current_time = max(current_time, range_end)
            
            # If we've covered the target period, break
            if current_time >= target_end:
                break
        
        # Check if there's a gap at the end
        if current_time < target_end:
            missing_periods.append((current_time, target_end))
        
        return missing_periods
    
    async def fetch_klines_data(self, session: aiohttp.ClientSession, symbol: str, interval: str, 
                               start_time: int, end_time: int, limit: int = 1000) -> List[List]:
        """Fetch klines data from Binance API."""
        url = f"{self.base_url}{self.klines_endpoint}"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    print(f"  ❌ API Error: {response.status} - {await response.text()}")
                    return []
        except Exception as e:
            print(f"  ❌ Request Error: {e}")
            return []
    
    async def download_period_data(self, start_date: datetime, end_date: datetime, batch_id: str) -> Dict[str, Any]:
        """Download data for a specific period using Binance's direct API."""
        print(f"📥 Downloading {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Convert to milliseconds
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        print(f"  🎯 Target: {since} to {until} ({(until - since) / (1000 * 60 * 60 * 24):.1f} days)")
        print(f"  📅 Start date: {start_date} ({since})")
        print(f"  📅 End date: {end_date} ({until})")
        
        all_data = []
        current_since = since
        batch_count = 0
        max_batches = 10000  # Safety limit
        consecutive_empty_batches = 0
        
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            while current_since < until and batch_count < max_batches:
                try:
                    # Fetch klines data (1000 records max per request)
                    print(f"  🔍 Requesting data from {current_since} ({datetime.fromtimestamp(current_since/1000)})")
                    
                    ohlcv = await self.fetch_klines_data(
                        session, 'ETHUSDT', '1m', 
                        current_since, until, 1000
                    )
                    
                    if not ohlcv:
                        consecutive_empty_batches += 1
                        print(f"  ⚠️ Empty batch {batch_count + 1} (consecutive: {consecutive_empty_batches})")
                        
                        if consecutive_empty_batches >= 3:
                            print(f"  ⚠️ No more data available after {consecutive_empty_batches} empty batches")
                            break
                        
                        # Move forward by 1 day and try again
                        current_since += 24 * 60 * 60 * 1000  # 1 day in milliseconds
                        await asyncio.sleep(2)
                        continue
                    
                    # Reset consecutive empty batches counter
                    consecutive_empty_batches = 0
                    
                    # Debug: Show what data we're getting
                    if ohlcv:
                        first_timestamp = ohlcv[0][0]
                        last_timestamp = ohlcv[-1][0]
                        first_date = datetime.fromtimestamp(first_timestamp/1000)
                        last_date = datetime.fromtimestamp(last_timestamp/1000)
                        print(f"  📊 Batch {batch_count + 1}: Downloaded {len(ohlcv)} records")
                        print(f"    📅 Data range: {first_date} to {last_date}")
                        print(f"    🎯 Requested: {datetime.fromtimestamp(current_since/1000)}")
                    
                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 60000  # Move to next minute
                    batch_count += 1
                    
                    print(f"  📊 Total so far: {len(all_data)} records")
                    
                    # Calculate progress
                    progress = (current_since - since) / (until - since) * 100
                    if batch_count % 5 == 0 or progress > 0:
                        print(f"  📈 Progress: {progress:.1f}% ({len(all_data)} records)")
                    
                    # Rate limiting - wait between requests
                    await asyncio.sleep(0.1)  # 100ms between requests
                    
                except Exception as e:
                    print(f"  ❌ Error in batch {batch_count}: {e}")
                    # Wait longer on error and try to continue
                    await asyncio.sleep(5)
                    consecutive_empty_batches += 1
                    if consecutive_empty_batches >= 3:
                        break
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Keep only the columns we need
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Check if we got enough data for the period
            expected_days = (end_date - start_date).days
            actual_days = (df.index.max() - df.index.min()).days
            expected_records = expected_days * 24 * 60  # 1-minute intervals
            actual_records = len(df)
            
            print(f"  📊 Data validation:")
            print(f"    Expected: {expected_days} days, {expected_records:,} records")
            print(f"    Actual: {actual_days} days, {actual_records:,} records")
            print(f"    Coverage: {actual_records / expected_records * 100:.1f}%")
            
            # Save to parquet in raw directory (monthly files)
            year_month = start_date.strftime('%Y_%m')
            filename = f"ethusdt_1m_{year_month}.parquet"
            filepath = self.raw_dir / filename
            df.to_parquet(filepath)
            
            print(f"  ✅ Saved {len(df)} records to {filename}")
            print(f"  📅 Date range: {df.index.min()} to {df.index.max()}")
            
            return {
                'success': True,
                'records': len(df),
                'file': filename,
                'start': df.index.min(),
                'end': df.index.max(),
                'coverage_percentage': actual_records / expected_records * 100
            }
        else:
            return {
                'success': False,
                'error': 'No data downloaded'
            }

async def main():
    """Main download function."""
    
    # Calculate date range: 4 years ago to 3 days ago
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=4*365)  # 4 years ago
    
    print(f"🎯 Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📅 Duration: {(end_date - start_date).days} days")
    
    # Initialize downloader
    downloader = BinanceHistoricalDownloader()
    
    # Analyze existing data
    print("\n📊 Analyzing existing raw data...")
    analysis = downloader.analyze_existing_raw_data()
    
    # Identify missing periods
    print("\n🔍 Identifying missing periods...")
    missing_periods = downloader.identify_missing_periods(start_date, end_date)
    
    if not missing_periods:
        print("✅ No missing data found! All data is already present.")
        return
    
    print(f"\n📋 Found {len(missing_periods)} missing periods:")
    total_missing_days = 0
    for i, (start, end) in enumerate(missing_periods, 1):
        duration = end - start
        days = duration.days
        total_missing_days += days
        print(f"  {i}. {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} ({days} days)")
    
    # Estimate download size
    estimated_records = total_missing_days * 24 * 60  # 1-minute intervals
    estimated_size_mb = estimated_records * 0.001  # Rough estimate: 1KB per record
    print(f"\n💾 Estimated download:")
    print(f"  Records needed: {estimated_records:,}")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB")
    print(f"  Estimated time: {estimated_records / 1000 * 0.1 / 60:.1f} minutes")
    
    # Download missing data for each period (no confirmation needed)
    print(f"\n🚀 Starting download of {len(missing_periods)} missing periods...")
    total_downloaded = 0
    successful_downloads = 0
    
    for i, (period_start, period_end) in enumerate(missing_periods, 1):
        print(f"\n📥 Downloading period {i}/{len(missing_periods)}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
        
        batch_id = f"batch_{i:03d}_{period_start.strftime('%Y%m%d')}"
        
        try:
            result = await downloader.download_period_data(period_start, period_end, batch_id)
            
            if result['success']:
                print(f"✅ Successfully downloaded {result['records']} records")
                total_downloaded += result['records']
                successful_downloads += 1
            else:
                print(f"❌ Failed to download period {i}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error downloading period {i}: {e}")
    
    print(f"\n🎉 Download complete!")
    print(f"  Successful downloads: {successful_downloads}/{len(missing_periods)}")
    print(f"  Total records downloaded: {total_downloaded:,}")

if __name__ == "__main__":
    print("🚀 Starting Binance Historical Data Download...")
    print("📅 Target: 4 years ago to 3 days ago")
    print("⏰ Interval: 1-minute")
    print("🏦 Exchange: Binance (Public API)")
    print("🔑 No API keys required")
    print("=" * 50)
    
    asyncio.run(main())
