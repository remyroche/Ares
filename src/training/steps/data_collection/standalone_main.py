#!/usr/bin/env python3
"""
Standalone Data Collection Pipeline Main

This module provides a standalone main function for the enhanced data collection pipeline
that doesn't depend on complex existing infrastructure.
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the standalone enhanced pipeline directly
from src.training.steps.data_collection.standalone_enhanced_pipeline import run_standalone_enhanced_data_collection_pipeline

async def main():
    """Main function to run data collection pipeline."""
    print("🚀 Step 1: Enhanced Data Collection Pipeline")
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
        result = await run_standalone_enhanced_data_collection_pipeline(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            config=config
        )
        success = result.get("success", False)
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED DATA COLLECTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All data collection steps completed:")
            print("   ✅ Raw data collection from exchange")
            print("   ✅ Data quality validation")
            print("   ✅ Data formatting and preprocessing")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print(f"📊 Pipeline ID: {result.get('pipeline_id', 'N/A')}")
            print(f"📈 Steps completed: {result.get('steps_completed', 0)}/{result.get('total_steps', 0)}")
            print(f"⚠️ Warnings: {len(result.get('warnings', []))}")
            print(f"❌ Errors: {len(result.get('errors', []))}")
            
            if result.get('warnings'):
                print("\n⚠️ Warnings:")
                for warning in result['warnings']:
                    print(f"   • {warning}")
            
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
                    'success': True,
                    'pipeline_result': result
                }, f, indent=2, default=str)
            
            print(f"💾 Configuration saved to: {config_file}")
            
        else:
            print("\n❌ ENHANCED DATA COLLECTION FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            
            if result.get('errors'):
                print("\n❌ Errors:")
                for error in result['errors']:
                    print(f"   • {error}")
            
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED DATA COLLECTION FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the data collection pipeline
    asyncio.run(main())