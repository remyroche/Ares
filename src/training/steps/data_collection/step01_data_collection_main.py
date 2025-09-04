#!/usr/bin/env python3
"""Step 1: Data Collection Pipeline.

This module provides the main interface for data collection with:
1. Raw data collection from exchanges
2. Data quality validation
3. Unified data loading
4. Data conversion and preprocessing
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_collection import run_data_collection_pipeline

async def main():
    """Main function to run data collection pipeline."""
    print("🚀 Step 1: Data Collection Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Data collection parameters
    config = {
        'force_rerun': True,
        'quality_checks': True,
        'validate_data': True,
        'convert_format': True,
        'random_state': 42,
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Force rerun: {config['force_rerun']}")
    print(f"   Quality checks: {config['quality_checks']}")
    print("=" * 80)
    
    # Run data collection pipeline
    start_time = time.time()
    
    try:
        success = await run_data_collection_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 DATA COLLECTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All data collection steps completed:")
            print("   ✅ Raw data collection from exchange")
            print("   ✅ Data quality validation")
            print("   ✅ Unified data loading")
            print("   ✅ Data conversion and preprocessing")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"data_collection_config_{symbol}_{timeframe}.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': config,
                    'execution_time': total_time,
                    'success': True
                }, f, indent=2)
            
            print(f"💾 Configuration saved to: {config_file}")
            
        else:
            print("\n❌ DATA COLLECTION FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 DATA COLLECTION FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the data collection pipeline
    asyncio.run(main())