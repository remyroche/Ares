#!/usr/bin/env python3
"""Step 16: Optimization Pipeline.

This module provides the main interface for optimization with:
1. Confidence calibration
2. Final parameters optimization
3. Parameter optimization wrapper
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.optimisation import run_optimisation_pipeline

async def main():
    """Main function to run optimization pipeline."""
    print("🚀 Step 16: Optimization Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Optimization parameters
    config = {
        'force_rerun': True,
        'confidence_calibration': True,
        'parameter_optimization': True,
        'random_state': 42,
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Confidence calibration: {config['confidence_calibration']}")
    print(f"   Parameter optimization: {config['parameter_optimization']}")
    print("=" * 80)
    
    # Run optimization pipeline
    start_time = time.time()
    
    try:
        success = await run_optimisation_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All optimization steps completed:")
            print("   ✅ Confidence calibration")
            print("   ✅ Final parameters optimization")
            print("   ✅ Parameter optimization wrapper")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"optimisation_config_{symbol}_{timeframe}.json"
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
            print("\n❌ OPTIMIZATION FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 OPTIMIZATION FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the optimization pipeline
    asyncio.run(main())