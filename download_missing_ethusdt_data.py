#!/usr/bin/env python3
"""
Download Missing ETHUSDT 1-minute Data

This script analyzes existing data in historical_data/binance/ethusdt/klines/
and downloads missing 1-minute ETHUSDT data from 4 years ago to 3 days ago.
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
from src.trading.execution.exchange_interface import create_exchange_interface
from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    KlinesDataProcessingPipeline, PipelineConfig, ResamplingConfig
)

logger = system_logger.getChild('MissingDataDownloader')

class MissingDataAnalyzer:
    """Analyzes existing data and identifies missing periods."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.exchange_dir = self.data_dir / "binance" / "ethusdt" / "klines"
        
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
                if not df.empty and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    date_ranges.append({
                        'file': file.name,
                        'start': df['timestamp'].min(),
                        'end': df['timestamp'].max(),
                        'records': len(df)
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

async def download_missing_data():
    """Download missing ETHUSDT 1-minute data."""
    
    # Calculate date range: 4 years ago to 3 days ago
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=4*365)  # 4 years ago
    
    print(f"🎯 Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Analyze existing data
    analyzer = MissingDataAnalyzer()
    print("📊 Analyzing existing data...")
    analysis = analyzer.analyze_existing_data()
    
    print(f"📁 Found {analysis['total_files']} existing files")
    print(f"📈 Analyzed {analysis['analyzed_files']} files successfully")
    
    # Identify missing periods
    print("🔍 Identifying missing periods...")
    missing_periods = analyzer.identify_missing_periods(start_date, end_date)
    
    if not missing_periods:
        print("✅ No missing data found! All data is already present.")
        return
    
    print(f"📋 Found {len(missing_periods)} missing periods:")
    for i, (start, end) in enumerate(missing_periods, 1):
        duration = end - start
        print(f"  {i}. {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} ({duration.days} days)")
    
    # Create exchange interface
    print("🔌 Setting up exchange connection...")
    exchange_config = {
        'exchange_type': 'binance',
        'api_key': '',  # Add your API key here
        'api_secret': '',  # Add your API secret here
        'testnet': True
    }
    
    exchange_interface = create_exchange_interface(exchange_config)
    await exchange_interface.connect()
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        data_dir="historical_data",
        exchange="binance",
        enable_logging=True,
        enable_gap_filling=True,
        enable_resampling=False,  # We only want 1m data
        enable_duplicate_handling=True,
        enable_quality_validation=True,
        batch_compatible=True
    )
    
    # Download missing data for each period
    total_downloaded = 0
    for i, (period_start, period_end) in enumerate(missing_periods, 1):
        print(f"\n📥 Downloading period {i}/{len(missing_periods)}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
        
        try:
            # Calculate how many days to download
            days_to_download = (period_end - period_start).days
            
            # Create pipeline for this period
            pipeline = KlinesDataProcessingPipeline(pipeline_config)
            
            # Download data for this period
            result = await pipeline.process_klines_data(
                symbol="ETHUSDT",
                interval="1m",
                years=min(days_to_download / 365, 1),  # Convert days to years, max 1 year per batch
                exchange_interface=exchange_interface,
                batch_id=f"missing_data_{i:03d}"
            )
            
            if result['pipeline_success']:
                print(f"✅ Successfully downloaded {result.get('records_downloaded', 0)} records")
                total_downloaded += result.get('records_downloaded', 0)
            else:
                print(f"❌ Failed to download period {i}: {result.get('errors', [])}")
                
        except Exception as e:
            print(f"❌ Error downloading period {i}: {e}")
    
    print(f"\n🎉 Download complete! Total records downloaded: {total_downloaded}")
    
    # Disconnect
    await exchange_interface.disconnect()

if __name__ == "__main__":
    print("🚀 Starting missing ETHUSDT data download...")
    print("📅 Target: 4 years ago to 3 days ago")
    print("⏰ Interval: 1-minute")
    print("🏦 Exchange: Binance")
    print("=" * 50)
    
    asyncio.run(download_missing_data())
