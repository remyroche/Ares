#!/usr/bin/env python3
"""
Quick test of SR quality data collection with new settings
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector

async def test_collection():
    """Test data collection with new 24-month + 12h sampling settings."""
    
    print("="*80)
    print("TESTING DATA COLLECTION WITH NEW SETTINGS")
    print("="*80)
    
    collector = SRQualityDataCollector()
    
    # Use last 24 months
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=730)
    
    print(f"\n📅 Collection Parameters:")
    print(f"   Period: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")
    print(f"   Symbol: ETHUSDT")
    print(f"   Exchange: binance")
    print(f"   Timeframe: 15m")
    print(f"   Sample freq: 0.5 days (12 hours)")
    print(f"   Forward days: 10")
    print(f"   Target samples: 5,000")
    
    print(f"\n🔄 Starting collection...")
    
    training_df = await collector.collect_training_data(
        symbol='ETHUSDT',
        exchange='binance',
        start_date=start_dt.strftime('%Y-%m-%d'),
        end_date=end_dt.strftime('%Y-%m-%d'),
        timeframe='15m',
        forward_days=10,
        sample_freq_days=0.5
    )
    
    print(f"\n✅ Collection Complete!")
    print(f"   Total samples: {len(training_df)}")
    print(f"   Date range: {training_df['date'].min()} → {training_df['date'].max()}")
    print(f"   Days covered: {(training_df['date'].max() - training_df['date'].min()).days}")
    print(f"   Unique dates: {training_df['date'].nunique()}")
    
    # Check features
    feature_cols = [c for c in training_df.columns if c.startswith('feature_')]
    print(f"\n🔧 Features: {len(feature_cols)}")
    print(f"   Sample:Feature ratio: {len(training_df)}:{len(feature_cols)} = {len(training_df)/len(feature_cols):.1f}:1")
    
    # Save
    saved_path = collector.save_training_data(training_df)
    print(f"\n💾 Saved to: {saved_path}")
    
    return training_df

if __name__ == "__main__":
    asyncio.run(test_collection())

