#!/usr/bin/env python3
"""
Binance Data Collection Script - Direct Klines Adapter Approach
Collects 4 years of historical klines data using the Binance Klines Adapter directly.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.pipeline_standards import PipelineStandards


async def main():
    """Run Binance data collection for 4 years using klines adapter directly."""
    
    print("=" * 80)
    print("🚀 BINANCE DATA COLLECTION - 4 YEARS (DIRECT ADAPTER)")
    print("=" * 80)
    print()
    
    # Configuration
    EXCHANGE = "binance"
    SYMBOL = "ETHUSDT"
    INTERVAL = "1m"
    YEARS = 4
    
    print(f"📊 Configuration:")
    print(f"   - Exchange: {EXCHANGE}")
    print(f"   - Symbol: {SYMBOL}")
    print(f"   - Interval: {INTERVAL}")
    print(f"   - Lookback: {YEARS} years")
    print()
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=YEARS * 365)
    
    print(f"📅 Time Range:")
    print(f"   - Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   - End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize Binance Klines Adapter
        print(f"🔧 Initializing Binance Klines Adapter...")
        adapter = BinanceKlinesAdapter(
            api_key=None,  # Not required for public data
            secret_key=None  # Not required for public data
        )
        print(f"✅ Adapter initialized")
        print()
        
        # Download data
        print(f"⏬ Downloading historical data...")
        print(f"   This may take several minutes...")
        print()
        
        all_data = []
        batch_size = 1000
        
        current_start = start_time
        batch_num = 1
        
        while current_start < end_time:
            # Calculate batch end time (max 1000 candles at 1m = ~16.67 hours)
            batch_end = min(current_start + timedelta(minutes=batch_size), end_time)
            
            print(f"   📦 Batch {batch_num}: {current_start.strftime('%Y-%m-%d %H:%M')} to {batch_end.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                # Fetch klines for this batch
                df_batch = await adapter.get_klines_data(
                    symbol=SYMBOL,
                    interval=INTERVAL,
                    start_time=current_start,
                    end_time=batch_end,
                    limit=batch_size
                )
                
                if df_batch is not None and len(df_batch) > 0:
                    all_data.append(df_batch)
                    total_candles = sum(len(df) for df in all_data)
                    print(f"      ✅ Received {len(df_batch)} candles (Total: {total_candles})")
                else:
                    print(f"      ⚠️  No data received for this batch")
                
            except Exception as e:
                print(f"      ❌ Error fetching batch: {e}")
            
            # Move to next batch
            current_start = batch_end
            batch_num += 1
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        print()
        
        if len(all_data) == 0:
            print("❌ No data collected. Exiting.")
            return 1
        
        # Concatenate all batches
        print(f"🔄 Combining all batches...")
        df = pd.concat(all_data, ignore_index=True)
        print(f"✅ DataFrame created: {df.shape}")
        print()
        
        # Display sample
        print(f"📊 Sample Data:")
        print(df.head())
        print()
        
        # Save to parquet
        print(f"💾 Saving to parquet...")
        
        # Use PipelineStandards for path
        standards = PipelineStandards()
        base_path = standards.build_path(
            path_type='raw_data',
            exchange=EXCHANGE,
            asset=SYMBOL.lower()
        )
        
        # Create output directory
        output_dir = Path(base_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        output_file = output_dir / f"klines_{EXCHANGE}_{SYMBOL}_{INTERVAL}_{YEARS}y_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"✅ Data saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        
        # Summary
        print("=" * 80)
        print("✅ DATA COLLECTION COMPLETED")
        print("=" * 80)
        print()
        print(f"📊 Summary:")
        print(f"   - Total Candles: {len(df)}")
        print(f"   - Data Shape: {df.shape}")
        print(f"   - Columns: {', '.join(df.columns.tolist())}")
        print(f"   - Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   - Output File: {output_file}")
        print()
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR IN DATA COLLECTION")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

