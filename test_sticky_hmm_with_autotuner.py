#!/usr/bin/env python3
"""Test Sticky Finite HMM with auto-tuner."""

import asyncio
import sys

async def main():
    print("=" * 80)
    print("STICKY FINITE HMM - WITH AUTO-TUNER TEST")
    print("=" * 80)
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
        run_sticky_finite_hmm_step
    )
    
    # Configuration with auto-tuning enabled
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'execution_mode': 'full',
        
        # Enable auto-tuning
        'enable_auto_tuning': True,
        
        # Auto-tuning configuration
        'auto_tuning_config': {
            'use_hierarchical': True,
            'n_rounds': 2,
            'tpe_trials': 50,  # Reduced for faster testing
            'timeout': 1800,   # 30 minutes
            'cache_dir': None
        },
        
        # Base parameters (will be optimized)
        'sticky_finite_hmm_params': {
            'K': 5,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'num_iters': 200,
            'lr': 1e-2,
            'pca_components': 15
        }
    }
    
    print(f"\n📋 Configuration:")
    print(f"   Symbol: {config['symbol']}")
    print(f"   Exchange: {config['exchange']}")
    print(f"   Timeframe: {config['regime_timeframe']}")
    print(f"   Auto-tuning: {config['enable_auto_tuning']}")
    print(f"   TPE trials: {config['auto_tuning_config']['tpe_trials']}")
    print(f"   Timeout: {config['auto_tuning_config']['timeout']}s")
    print()
    
    print("🚀 Executing Sticky Finite HMM with auto-tuner...")
    print("=" * 80)
    print()
    
    try:
        result = await run_sticky_finite_hmm_step(config)
        
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if result['success']:
            print(f"✅ SUCCESS!")
            print(f"   Execution time: {result['execution_time']:.2f}s")
            print(f"   Regimes discovered: {result.get('n_regimes', 0)}")
            print(f"   Composite score: {result.get('composite_score', 0.0):.3f}")
            
            # Check for auto-tuning results
            if 'auto_tuning_results' in result:
                print(f"\n🎯 Auto-tuning Results:")
                tuning = result['auto_tuning_results']
                print(f"   Best score: {tuning.get('best_score', 0.0):.4f}")
                print(f"   Best params: {tuning.get('best_params', {})}")
                print(f"   Trials completed: {tuning.get('n_trials', 0)}")
                print(f"   Optimization time: {tuning.get('optimization_time', 0.0):.2f}s")
            
            # Check generated reports
            from pathlib import Path
            outcomes_dir = Path("outcomes") / "sticky_finite_hmm_clustering" / "ETHUSDT" / "binance" / "1h"
            
            if outcomes_dir.exists():
                csv_files = list(outcomes_dir.glob("*.csv"))
                
                print(f"\n📊 Generated Reports:")
                print(f"   Location: {outcomes_dir}")
                print(f"   CSV files: {len(csv_files)}")
                for f in sorted(csv_files)[-3:]:  # Show last 3
                    size_kb = f.stat().st_size / 1024
                    print(f"      - {f.name} ({size_kb:.1f} KB)")
                
                # Check for economic metrics in latest CSV
                if csv_files:
                    import pandas as pd
                    latest = sorted(csv_files)[-1]
                    if 'all_results' in latest.name:
                        df = pd.read_csv(latest)
                        sharpe_cols = [c for c in df.columns if 'sharpe' in c.lower() and 'regime_' in c]
                        economic_cols = [c for c in df.columns if any(x in c for x in ['mean_return', 'volatility', 'win_rate', 'profit_factor'])]
                        
                        print(f"\n   📈 Economic metrics in CSV:")
                        print(f"      Total columns: {len(df.columns)}")
                        print(f"      Sharpe columns: {len(sharpe_cols)}")
                        print(f"      Economic columns: {len(economic_cols)}")
                        
                        if sharpe_cols:
                            print(f"      Sample: {sharpe_cols[:3]}")
            
            print("\n" + "=" * 80)
            print("🎉 TEST COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            return 0
            
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

