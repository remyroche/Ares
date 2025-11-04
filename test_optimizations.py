#!/usr/bin/env python3
"""
Test Sticky Finite HMM optimizations and measure speedup.
"""

import asyncio
import pandas as pd
import time

async def test_without_posteriors():
    """Test with posteriors DISABLED (auto-tuning mode)."""
    print("\n" + "="*80)
    print("TEST 1: WITHOUT POSTERIORS (Auto-Tuning Mode)")
    print("="*80)
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import run_sticky_finite_hmm_step
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'execution_mode': 'full',
        'sticky_finite_hmm_params': {
            'K': 5,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'num_iters': 200,  # Same as before
            'lr': 1e-2,
            'pca_components': 15,
            'compute_posteriors': False  # FAST MODE
        }
    }
    
    start = time.time()
    result = await run_sticky_finite_hmm_step(config)
    elapsed = time.time() - start
    
    if result['success']:
        print(f"\n✅ SUCCESS (No Posteriors)")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Regimes: {result.get('n_regimes', 0)}")
        print(f"   Score: {result.get('composite_score', 0.0):.3f}")
        return elapsed
    else:
        print(f"\n❌ FAILED: {result.get('error')}")
        return None


async def test_with_posteriors():
    """Test with posteriors ENABLED (production mode)."""
    print("\n" + "="*80)
    print("TEST 2: WITH POSTERIORS (Production Mode)")
    print("="*80)
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import run_sticky_finite_hmm_step
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'execution_mode': 'full',
        'sticky_finite_hmm_params': {
            'K': 5,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'num_iters': 200,  # Same as before
            'lr': 1e-2,
            'pca_components': 15,
            'compute_posteriors': True  # FULL MODE
        }
    }
    
    start = time.time()
    result = await run_sticky_finite_hmm_step(config)
    elapsed = time.time() - start
    
    if result['success']:
        print(f"\n✅ SUCCESS (With Posteriors)")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Regimes: {result.get('n_regimes', 0)}")
        print(f"   Score: {result.get('composite_score', 0.0):.3f}")
        
        # Check CSV for Sharpe ratios
        from pathlib import Path
        csv_path = Path("outcomes/sticky_finite_hmm_clustering/ETHUSDT/binance/1h")
        csv_files = sorted(csv_path.glob("sticky_finite_hmm_all_results_*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[-1])
            sharpe_cols = [c for c in df.columns if 'sharpe' in c.lower() and 'regime_' in c]
            print(f"   Sharpe columns: {len(sharpe_cols)}")
            if sharpe_cols:
                for col in sharpe_cols[:3]:
                    print(f"      {col}: {df[col].values[0]:.4f}")
        
        return elapsed
    else:
        print(f"\n❌ FAILED: {result.get('error')}")
        return None


async def main():
    print("\n" + "🚀"*40)
    print("STICKY FINITE HMM - OPTIMIZATION BENCHMARK")
    print("🚀"*40)
    
    # Test 1: Without posteriors (fast)
    time_without = await test_without_posteriors()
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Test 2: With posteriors (full)
    time_with = await test_with_posteriors()
    
    # Results
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80)
    
    if time_without and time_with:
        speedup = time_with / time_without
        time_saved = time_with - time_without
        
        print(f"\n⏱️  Without Posteriors: {time_without:.2f}s")
        print(f"⏱️  With Posteriors:    {time_with:.2f}s")
        print(f"⚡ Time Saved:         {time_saved:.2f}s")
        print(f"🚀 Speedup Factor:     {speedup:.2f}x")
        print(f"\n💡 Auto-tuning with 100 trials:")
        print(f"   Old time: {time_with * 100 / 60:.1f} minutes")
        print(f"   New time: {time_without * 100 / 60:.1f} minutes")
        print(f"   Saved:    {time_saved * 100 / 60:.1f} minutes")
        
        print("\n" + "="*80)
        if speedup > 1.3:
            print("✅ OPTIMIZATION SUCCESSFUL! (>30% faster)")
        else:
            print("⚠️  Modest improvement (<30% faster)")
        print("="*80)
    else:
        print("\n❌ Benchmark incomplete")
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

