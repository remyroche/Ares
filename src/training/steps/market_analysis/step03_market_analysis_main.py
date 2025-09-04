#!/usr/bin/env python3
"""Step 3: Market Analysis Pipeline.

This module provides the main interface for market analysis with:
1. HMM regime discovery and clustering
2. Regime data splitting and labeling
3. Feature engineering and selection
4. Advanced matrix operations
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis import run_market_analysis_pipeline

async def main():
    """Main function to run market analysis pipeline."""
    print("🚀 Step 3: Market Analysis Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Market analysis parameters
    config = {
        'force_rerun': True,
        'hmm_clustering': True,
        'regime_splitting': True,
        'feature_engineering': True,
        'matrix_operations': True,
        'feature_selection': True,
        'random_state': 42,
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   HMM clustering: {config['hmm_clustering']}")
    print(f"   Regime splitting: {config['regime_splitting']}")
    print(f"   Feature engineering: {config['feature_engineering']}")
    print("=" * 80)
    
    # Run market analysis pipeline
    start_time = time.time()
    
    try:
        success = await run_market_analysis_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 MARKET ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All market analysis steps completed:")
            print("   ✅ HMM regime discovery and clustering")
            print("   ✅ Regime data splitting and labeling")
            print("   ✅ Feature engineering and selection")
            print("   ✅ Advanced matrix operations")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"market_analysis_config_{symbol}_{timeframe}.json"
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
            print("\n❌ MARKET ANALYSIS FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 MARKET ANALYSIS FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the market analysis pipeline
    asyncio.run(main())