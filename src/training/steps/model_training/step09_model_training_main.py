#!/usr/bin/env python3
"""Step 9: Model Training Pipeline.

This module provides the main interface for model training with:
1. HMM-based training
2. Unified regime intelligence
3. Analyst creation and enhancement
4. Tactician specialist training
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.model_training import run_model_training_pipeline

async def main():
    """Main function to run model training pipeline."""
    print("🚀 Step 9: Model Training Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Model training parameters
    config = {
        'force_rerun': True,
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'random_state': 42,
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   HMM training: {config['hmm_training']}")
    print(f"   Regime intelligence: {config['regime_intelligence']}")
    print(f"   Analyst creation: {config['analyst_creation']}")
    print(f"   Tactician training: {config['tactician_training']}")
    print("=" * 80)
    
    # Run model training pipeline
    start_time = time.time()
    
    try:
        success = await run_model_training_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All model training steps completed:")
            print("   ✅ HMM-based training")
            print("   ✅ Unified regime intelligence")
            print("   ✅ Analyst creation and enhancement")
            print("   ✅ Ensemble creation")
            print("   ✅ Tactician specialist training")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"model_training_config_{symbol}_{timeframe}.json"
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
            print("\n❌ MODEL TRAINING FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 MODEL TRAINING FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the model training pipeline
    asyncio.run(main())