#!/usr/bin/env python3
"""
Download ETHUSDT 1-minute Data from Binance

This script downloads missing ETHUSDT 1-minute data from 4 years ago to 3 days ago.
It will only download data that isn't already present.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import system_logger

logger = system_logger.getChild('ETHUSDTDownloader')

class ETHUSDTDataDownloader:
    """Downloads missing ETHUSDT 1-minute data from Binance."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.exchange_dir = self.data_dir / "binance" / "ethusdt" / "klines"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_existing_data(self) -> Dict[str, Any]:
        """Analyze existing data files and return coverage information."""
        if not self.exchange_dir.exists():
            return {
                "total_files": 0,
                "date_ranges": [],
                "missing_periods": [],
                "coverage_percentage": 0.0
            }
        
        # Find all parquet files
        parquet_files = list(self.exchange_dir.glob("*.parquet"))
        
        date_ranges = []
        for file in parquet_files:
            try:
                # Load the file to get date range
                df = pd.read_parquet(file)
                if not df.empty:
                    # Try different timestamp column names
                    timestamp_col = None
                    for col in ['timestamp', 'time', 'datetime', 'date']:
                        if col in df.columns:
                            timestamp_col = col
                            break
                    
                    if timestamp_col:
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                        # Only include files with reasonable timestamps (not 1970)
                        if df[timestamp_col].min().year > 2020:
                            date_ranges.append({
                                'file': file.name,
                                'start': df[timestamp_col].min(),
                                'end': df[timestamp_col].max(),
                                'records': len(df),
                                'interval': '1m' if '1m' in file.name else 'unknown'
                            })
            except Exception as e:
                logger.warning(f"Could not analyze file {file}: {e}")
        
        # Sort by start date
        date_ranges.sort(key=lambda x: x['start'])
        
        return {
            "total_files": len(parquet_files),
            "date_ranges": date_ranges,
            "analyzed_files": len(date_ranges)
        }
    
    def identify_missing_periods(self, target_start: datetime, target_end: datetime) -> List[Tuple[datetime, datetime]]:
        """Identify missing periods between target_start and target_end."""
        analysis = self.analyze_existing_data()
        date_ranges = analysis['date_ranges']
        
        if not date_ranges:
            # No existing data, need to download everything
            return [(target_start, target_end)]
        
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
    
    async def download_period_data(self, start_date: datetime, end_date: datetime, batch_id: str) -> Dict[str, Any]:
        """Download data for a specific period using Binance API."""
        try:
            import ccxt
            
            # Initialize Binance exchange
            exchange = ccxt.binance({
                'apiKey': '',  # Add your API key here
                'secret': '',  # Add your API secret here
                'sandbox': True,  # Use testnet
                'enableRateLimit': True,
            })
            
            print(f"📥 Downloading {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Convert to milliseconds
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    # Fetch klines data (1000 records max per request)
                    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1m', since=current_since, limit=1000)
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 60000  # Move to next minute
                    
                    print(f"  📊 Downloaded {len(ohlcv)} records, total: {len(all_data)}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"  ❌ Error fetching data: {e}")
                    break
            
            if all_data:
                # Convert to DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Save to parquet
                filename = f"klines_binance_ETHUSDT_1m_{batch_id}.parquet"
                filepath = self.exchange_dir / filename
                df.to_parquet(filepath)
                
                print(f"  ✅ Saved {len(df)} records to {filename}")
                
                return {
                    'success': True,
                    'records': len(df),
                    'file': filename,
                    'start': df.index.min(),
                    'end': df.index.max()
                }
            else:
                return {
                    'success': False,
                    'error': 'No data downloaded'
                }
                
        except Exception as e:
            print(f"  ❌ Error downloading period: {e}")
            return {
                'success': False,
                'error': str(e)
            }

async def main():
    """Main download function."""
    
    # Calculate date range: 4 years ago to 3 days ago
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=4*365)  # 4 years ago
    
    print(f"🎯 Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📅 Duration: {(end_date - start_date).days} days")
    
    # Initialize downloader
    downloader = ETHUSDTDataDownloader()
    
    # Analyze existing data
    print("\n📊 Analyzing existing data...")
    analysis = downloader.analyze_existing_data()
    
    print(f"📁 Found {analysis['total_files']} existing files")
    print(f"📈 Analyzed {analysis['analyzed_files']} files successfully")
    
    # Identify missing periods
    print("\n🔍 Identifying missing periods...")
    missing_periods = downloader.identify_missing_periods(start_date, end_date)
    
    if not missing_periods:
        print("✅ No missing data found! All data is already present.")
        return
    
    print(f"📋 Found {len(missing_periods)} missing periods:")
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
    
    # Ask for confirmation
    response = input(f"\n❓ Download {len(missing_periods)} missing periods? (y/N): ")
    if response.lower() != 'y':
        print("❌ Download cancelled.")
        return
    
    # Download missing data for each period
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
    print("🚀 Starting ETHUSDT data download...")
    print("📅 Target: 4 years ago to 3 days ago")
    print("⏰ Interval: 1-minute")
    print("🏦 Exchange: Binance")
    print("=" * 50)
    
    asyncio.run(main())
