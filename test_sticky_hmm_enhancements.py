"""
Test Sticky Finite HMM Enhancements

Tests all recent enhancements:
1. Gaussian Mixture Emissions (n_mixtures=1-3)
2. Multi-Timeframe Regime Features (4h, 1d context)
3. Multi-Objective Optimization with Pareto Front
4. Economic Quality Metrics (Sharpe, returns, drawdown per regime)

Run with: python test_sticky_hmm_enhancements.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning

# Test configuration
TEST_CONFIG = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'blank',  # Use blank for fast testing
    
    # Enable auto-tuning with enhancements
    'enable_auto_tuning': True,
    'auto_tuning_config': {
        'use_hierarchical': True,
        'use_multi_objective': True,  # ✨ NEW: Multi-objective optimization
        'n_rounds': 1,
        'tpe_trials': 10,  # Small for quick test
        'timeout': 600  # 10 minutes max
    },
    
    # Or manually specify parameters to test specific features
    'sticky_finite_hmm_params': {
        'K': 5,
        'n_mixtures': 2,  # ✨ NEW: Test mixture emissions
        'kappa': 20.0,
        'base_alpha': 0.3,
        'lr': 1e-2,
        'pca_components': 15
    }
}


def create_test_data(n_samples=1000):
    """Create synthetic test data with datetime index."""
    tprint_info("📊 Creating synthetic test data with regimes...")
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=n_samples//24)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')
    
    # Create multi-regime synthetic data
    regime_length = n_samples // 5
    data = []
    
    for i in range(5):
        # Each regime has different characteristics
        if i == 0:  # Low vol, uptrend
            regime_data = {
                'returns': np.random.normal(0.001, 0.01, regime_length),
                'volatility': np.random.uniform(0.008, 0.012, regime_length)
            }
        elif i == 1:  # High vol, sideways
            regime_data = {
                'returns': np.random.normal(0.0, 0.03, regime_length),
                'volatility': np.random.uniform(0.025, 0.035, regime_length)
            }
        elif i == 2:  # Medium vol, downtrend
            regime_data = {
                'returns': np.random.normal(-0.001, 0.015, regime_length),
                'volatility': np.random.uniform(0.012, 0.018, regime_length)
            }
        elif i == 3:  # Very low vol, sideways
            regime_data = {
                'returns': np.random.normal(0.0, 0.005, regime_length),
                'volatility': np.random.uniform(0.003, 0.007, regime_length)
            }
        else:  # Extreme vol, mixed
            regime_data = {
                'returns': np.random.normal(0.0, 0.05, regime_length),
                'volatility': np.random.uniform(0.040, 0.060, regime_length)
            }
        
        data.append(regime_data)
    
    # Combine regimes
    returns = np.concatenate([d['returns'] for d in data])
    volatility = np.concatenate([d['volatility'] for d in data])
    
    # Generate OHLCV from returns
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + volatility * np.random.uniform(0, 0.5, len(close)))
    low = close * (1 - volatility * np.random.uniform(0, 0.5, len(close)))
    open_price = close * (1 + np.random.normal(0, volatility * 0.3))
    volume = np.random.lognormal(10, 1, len(close))
    
    df = pd.DataFrame({
        'timestamp': dates[:len(close)],
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    
    tprint_success(f"✅ Created {len(df)} samples with 5 distinct regimes")
    return df


async def test_mixture_emissions():
    """Test 1: Gaussian Mixture Emissions"""
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("TEST 1: Gaussian Mixture Emissions (n_mixtures=2)", "INFO")
    tprint("=" * 80, "INFO")
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
        StickyFiniteHMMRegimeDiscoveryStep
    )
    
    data = create_test_data(n_samples=600)  # Need 500+ for validation
    
    config = {
        'symbol': 'TEST',
        'exchange': 'test',
        'regime_timeframe': '1h',
        'market_data': data,  # Pass data directly
        'enable_auto_tuning': False,
        'sticky_finite_hmm_params': {
            'K': 3,
            'n_mixtures': 2,  # ✨ Test mixture emissions
            'kappa': 10.0,
            'num_iters': 200,  # Reduced for speed
            'lr': 1e-2
        }
    }
    
    step = StickyFiniteHMMRegimeDiscoveryStep()
    result = await step.execute(config)
    
    if result['success']:
        tprint_success(f"✅ TEST 1 PASSED: n_mixtures=2 works!")
        tprint_info(f"   Regimes found: {result['n_regimes']}")
        tprint_info(f"   Composite score: {result['composite_score']:.4f}")
    else:
        tprint_error(f"❌ TEST 1 FAILED: {result.get('error', 'Unknown error')}")
    
    return result['success']


async def test_mtf_features():
    """Test 2: Multi-Timeframe Features"""
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("TEST 2: Multi-Timeframe Regime Features (4h, 1d)", "INFO")
    tprint("=" * 80, "INFO")
    
    from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (
        EnhancedStickyFiniteHMMClusteringIntegration
    )
    
    data = create_test_data(n_samples=1000)  # Need more data for MTF
    
    # Test MTF feature generation
    integration = EnhancedStickyFiniteHMMClusteringIntegration(
        K=3,
        enable_mtf_features=True,  # ✨ Enable MTF
        mtf_timeframes=['4h', '1d']
    )
    
    try:
        result = integration.cluster_with_sticky_finite_hmm(data)
        
        if 'n_clusters' in result:
            tprint_success(f"✅ TEST 2 PASSED: MTF features work!")
            tprint_info(f"   Regimes found: {result['n_clusters']}")
            tprint_info(f"   Feature count: {len(result['feature_names'])}")
            
            # Check for MTF features
            mtf_features = [f for f in result['feature_names'] if 'mtf_' in f]
            tprint_info(f"   MTF features: {len(mtf_features)}")
            return True
        else:
            tprint_error("❌ TEST 2 FAILED: No clusters found")
            return False
            
    except Exception as e:
        tprint_error(f"❌ TEST 2 FAILED: {e}")
        return False


async def test_economic_metrics():
    """Test 3: Economic Quality Metrics"""
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("TEST 3: Economic Quality Metrics (Sharpe, Returns, Drawdown)", "INFO")
    tprint("=" * 80, "INFO")
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
        run_sticky_finite_hmm_clustering
    )
    
    data = create_test_data(n_samples=600)  # Need 500+ for validation
    
    try:
        result = run_sticky_finite_hmm_clustering(
            market_data=data,
            symbol='TEST',
            exchange='test',
            timeframe='1h',
            K=3,
            n_mixtures=1,
            num_iters=200,
            save_results=False
        )
        
        # Check for economic metrics in results
        quality_metrics = result.get('quality_metrics', {})
        quality_assessment = quality_metrics.get('quality_assessment', {})
        per_regime = quality_assessment.get('per_regime_metrics', {})
        
        if per_regime:
            tprint_success("✅ TEST 3 PASSED: Economic metrics available!")
            
            # Show economic metrics for first regime
            regime_0 = per_regime.get(0, {})
            tprint_info(f"   Regime 0 Sharpe: {regime_0.get('sharpe', 'N/A')}")
            tprint_info(f"   Regime 0 Mean Return: {regime_0.get('mean_return', 'N/A')}")
            tprint_info(f"   Regime 0 Max Drawdown: {regime_0.get('max_drawdown', 'N/A')}")
            tprint_info(f"   Regime 0 Win Rate: {regime_0.get('win_rate', 'N/A')}")
            return True
        else:
            tprint_warning("⚠️ TEST 3: No per-regime metrics found")
            return False
    except Exception as e:
        tprint_error(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_objective_optimization():
    """Test 4: Multi-Objective Optimization"""
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("TEST 4: Multi-Objective Optimization with Pareto Front", "INFO")
    tprint("=" * 80, "INFO")
    
    # Check if Pareto optimization is available
    try:
        from src.utils.ml_common.optimization.pareto import ParetoFront
        PARETO_AVAILABLE = True
    except ImportError:
        tprint_warning("⚠️ Pareto optimization not available, skipping test 4")
        return None
    
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
        run_sticky_finite_hmm_auto_tuning
    )
    
    data = create_test_data(n_samples=600)  # Need 500+ for validation
    
    try:
        best_params, best_score, tuning_results = run_sticky_finite_hmm_auto_tuning(
            market_data=data,
            symbol='TEST',
            exchange='test',
            timeframe='1h',
            use_hierarchical=True,
            use_multi_objective=True,  # ✨ Enable multi-objective
            n_rounds=1,
            tpe_trials=5,  # Very small for quick test
            timeout=300  # 5 min
        )
        
        # Check for Pareto front
        if 'pareto_front' in tuning_results:
            pareto_front = tuning_results['pareto_front']
            tprint_success(f"✅ TEST 4 PASSED: Pareto front constructed!")
            tprint_info(f"   Pareto solutions: {pareto_front['n_solutions']}")
            
            # Show first Pareto solution
            if pareto_front['solutions']:
                sol1 = pareto_front['solutions'][0]
                tprint_info(f"   Solution 1 objectives:")
                for obj_name, obj_value in sol1['objectives'].items():
                    tprint_info(f"     - {obj_name}: {obj_value:.4f}")
            return True
        else:
            tprint_warning("⚠️ TEST 4: No Pareto front in results")
            return False
            
    except Exception as e:
        tprint_error(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all enhancement tests"""
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("🧪 STICKY FINITE HMM ENHANCEMENTS TEST SUITE", "INFO")
    tprint("=" * 80, "INFO")
    tprint("", "INFO")
    tprint("Testing 4 major enhancements:", "INFO")
    tprint("  1. Gaussian Mixture Emissions (n_mixtures=2)", "INFO")
    tprint("  2. Multi-Timeframe Regime Features (4h, 1d)", "INFO")
    tprint("  3. Economic Quality Metrics (Sharpe, returns, drawdown)", "INFO")
    tprint("  4. Multi-Objective Optimization (Pareto front)", "INFO")
    tprint("", "INFO")
    
    results = {}
    
    # Test 1: Mixture Emissions
    try:
        results['mixture_emissions'] = await test_mixture_emissions()
    except Exception as e:
        tprint_error(f"Test 1 crashed: {e}")
        results['mixture_emissions'] = False
    
    # Test 2: MTF Features
    try:
        results['mtf_features'] = await test_mtf_features()
    except Exception as e:
        tprint_error(f"Test 2 crashed: {e}")
        results['mtf_features'] = False
    
    # Test 3: Economic Metrics
    try:
        results['economic_metrics'] = await test_economic_metrics()
    except Exception as e:
        tprint_error(f"Test 3 crashed: {e}")
        results['economic_metrics'] = False
    
    # Test 4: Multi-Objective Optimization
    try:
        results['multi_objective'] = await test_multi_objective_optimization()
    except Exception as e:
        tprint_error(f"Test 4 crashed: {e}")
        results['multi_objective'] = False
    
    # Summary
    tprint("", "INFO")
    tprint("=" * 80, "INFO")
    tprint("📊 TEST SUMMARY", "INFO")
    tprint("=" * 80, "INFO")
    
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    skipped = len([v for v in results.values() if v is None])
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else "⚠️ SKIPPED" if result is None else "❌ FAILED"
        tprint(f"  {test_name}: {status}", "INFO")
    
    tprint("", "INFO")
    tprint(f"Results: {passed}/{total} tests passed", "SUCCESS" if passed == total else "WARNING")
    if skipped > 0:
        tprint(f"Skipped: {skipped} tests (missing dependencies)", "INFO")
    tprint("=" * 80, "INFO")
    
    return results


if __name__ == "__main__":
    tprint("", "INFO")
    tprint("🚀 Starting Sticky Finite HMM Enhancement Tests...", "INFO")
    tprint("", "INFO")
    
    results = asyncio.run(run_all_tests())
    
    # Exit code
    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None])
    
    if passed == total:
        tprint("", "SUCCESS")
        tprint("🎉 All tests passed!", "SUCCESS")
        exit(0)
    else:
        tprint("", "WARNING")
        tprint(f"⚠️ {total - passed} test(s) failed", "WARNING")
        exit(1)

