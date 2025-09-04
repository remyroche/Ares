#!/usr/bin/env python3
"""Step 18: Backtesting Pipeline.

This module provides the main interface for backtesting with:
1. Walk forward validation
2. Monte Carlo validation
3. A/B testing
4. Model saving and persistence
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.backtesting import run_backtesting_pipeline

async def main():
    """Main function to run backtesting pipeline."""
    print("🚀 Step 18: Backtesting Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Backtesting parameters
    config = {
        'force_rerun': True,
        'walk_forward_validation': True,
        'monte_carlo_validation': True,
        'ab_testing': True,
        'model_saving': True,
        'random_state': 42,
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Walk forward validation: {config['walk_forward_validation']}")
    print(f"   Monte Carlo validation: {config['monte_carlo_validation']}")
    print(f"   A/B testing: {config['ab_testing']}")
    print(f"   Model saving: {config['model_saving']}")
    print("=" * 80)
    
    # Run backtesting pipeline
    start_time = time.time()
    
    try:
        success = await run_backtesting_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 BACKTESTING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All backtesting steps completed:")
            print("   ✅ Walk forward validation")
            print("   ✅ Monte Carlo validation")
            print("   ✅ A/B testing")
            print("   ✅ Model saving and persistence")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"backtesting_config_{symbol}_{timeframe}.json"
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
            print("\n❌ BACKTESTING FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 BACKTESTING FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the backtesting pipeline
    asyncio.run(main())