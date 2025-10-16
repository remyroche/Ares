#!/usr/bin/env python3
"""
Simple Data Analyzer for ETHUSDT

Analyzes existing data in historical_data/binance/ethusdt/klines/
and shows what data is missing from 4 years ago to 3 days ago.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

def analyze_existing_data(data_dir: str = "historical_data") -> Dict[str, Any]:
    """Analyze existing data files and return coverage information."""
    data_path = Path(data_dir) / "binance" / "ethusdt" / "klines"
    
    if not data_path.exists():
        return {
            "total_files": 0,
            "date_ranges": [],
            "missing_periods": [],
            "coverage_percentage": 0.0
        }
    
    # Find all parquet files
    parquet_files = list(data_path.glob("*.parquet"))
    
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
                    date_ranges.append({
                        'file': file.name,
                        'start': df[timestamp_col].min(),
                        'end': df[timestamp_col].max(),
                        'records': len(df),
                        'interval': '1m' if '1m' in file.name else 'unknown'
                    })
        except Exception as e:
            print(f"Could not analyze file {file}: {e}")
    
    # Sort by start date
    date_ranges.sort(key=lambda x: x['start'])
    
    return {
        "total_files": len(parquet_files),
        "date_ranges": date_ranges,
        "analyzed_files": len(date_ranges)
    }

def identify_missing_periods(date_ranges: List[Dict], target_start: datetime, target_end: datetime) -> List[Tuple[datetime, datetime]]:
    """Identify missing periods between target_start and target_end."""
    
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

def main():
    """Main analysis function."""
    
    # Calculate date range: 4 years ago to 3 days ago
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=4*365)  # 4 years ago
    
    print(f"🎯 Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📅 Duration: {(end_date - start_date).days} days")
    
    # Analyze existing data
    print("\n📊 Analyzing existing data...")
    analysis = analyze_existing_data()
    
    print(f"📁 Found {analysis['total_files']} existing files")
    print(f"📈 Analyzed {analysis['analyzed_files']} files successfully")
    
    if analysis['date_ranges']:
        print(f"\n📋 Existing data periods:")
        for i, range_info in enumerate(analysis['date_ranges'][:10], 1):  # Show first 10
            print(f"  {i}. {range_info['start'].strftime('%Y-%m-%d %H:%M')} to {range_info['end'].strftime('%Y-%m-%d %H:%M')} ({range_info['records']} records, {range_info['interval']})")
        
        if len(analysis['date_ranges']) > 10:
            print(f"  ... and {len(analysis['date_ranges']) - 10} more periods")
    
    # Identify missing periods
    print("\n🔍 Identifying missing periods...")
    missing_periods = identify_missing_periods(analysis['date_ranges'], start_date, end_date)
    
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
    
    print(f"\n📊 Summary:")
    print(f"  Total target period: {(end_date - start_date).days} days")
    print(f"  Missing periods: {len(missing_periods)}")
    print(f"  Total missing days: {total_missing_days} days")
    print(f"  Coverage: {((end_date - start_date).days - total_missing_days) / (end_date - start_date).days * 100:.1f}%")
    
    # Estimate download size
    estimated_records = total_missing_days * 24 * 60  # 1-minute intervals
    estimated_size_mb = estimated_records * 0.001  # Rough estimate: 1KB per record
    print(f"\n💾 Estimated download:")
    print(f"  Records needed: {estimated_records:,}")
    print(f"  Estimated size: {estimated_size_mb:.1f} MB")

if __name__ == "__main__":
    print("🚀 Starting ETHUSDT data analysis...")
    print("📅 Target: 4 years ago to 3 days ago")
    print("⏰ Interval: 1-minute")
    print("🏦 Exchange: Binance")
    print("=" * 50)
    
    main()
