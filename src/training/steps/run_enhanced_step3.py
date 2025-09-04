#!/usr/bin/env python3
"""Integration script for Enhanced Step 3: HMM Regime Discovery.

This script demonstrates how to run the enhanced step 3 with all improvements:
1. Bayesian parameter optimization
2. Enhanced regime discovery features
3. Economic significance validation
4. Ensemble clustering (HMM + K-means + DBSCAN)
5. Enhanced ML transition detection (Random Forest + LGBM)
"""

import asyncio
import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.step03_enhanced_hmm_regime_discovery import run_enhanced_step

async def main():
    """Main function to run enhanced step 3."""
    print("🚀 Enhanced Step 3: HMM Regime Discovery with All Improvements")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Enhanced parameters
    enhanced_config = {
        # Bayesian optimization parameters
        'n_trials': 50,  # Number of Optuna trials
        'timeout_minutes': 15,  # Timeout for optimization
        'cv_folds': 3,  # Cross-validation folds
        'random_state': 42,
        
        # Ensemble clustering parameters
        'ensemble_weights': {
            'hmm': 0.4,
            'kmeans': 0.3,
            'dbscan': 0.3
        },
        
        # Enhanced ML transition detection parameters
        'initial_features': 20,  # Start with top 20 features
        'feature_increment': 10,  # Add 10 features at a time
        'max_features': 100,  # Maximum features to consider
        'min_improvement': 0.001,  # Minimum improvement threshold
        'patience': 3,  # Patience for early stopping
    }
    
    print(f"📊 Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Bayesian trials: {enhanced_config['n_trials']}")
    print(f"   Timeout: {enhanced_config['timeout_minutes']} minutes")
    print(f"   Initial features: {enhanced_config['initial_features']}")
    print(f"   Feature increment: {enhanced_config['feature_increment']}")
    print("=" * 80)
    
    # Run enhanced step 3
    start_time = time.time()
    
    try:
        success = await run_enhanced_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=True,
            **enhanced_config
        )
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED STEP 3 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ All improvements integrated:")
            print("   ✅ Bayesian parameter optimization with Optuna")
            print("   ✅ Enhanced regime discovery features")
            print("   ✅ Economic significance validation")
            print("   ✅ Ensemble clustering (HMM + K-means + DBSCAN)")
            print("   ✅ Enhanced ML transition detection (Random Forest + LGBM)")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
            # Save configuration for future reference
            config_file = Path(data_dir) / f"enhanced_step3_config_{symbol}_{timeframe}.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': enhanced_config,
                    'execution_time': total_time,
                    'success': True
                }, f, indent=2)
            
            print(f"💾 Configuration saved to: {config_file}")
            
        else:
            print("\n❌ ENHANCED STEP 3 FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED STEP 3 FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise

if __name__ == "__main__":
    # Run the enhanced step 3
    asyncio.run(main())