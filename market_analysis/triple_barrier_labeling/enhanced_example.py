"""
Enhanced Triple Barrier Labeling Example

This example demonstrates the enhanced optimized triple barrier labeling system
with matrix operations, coarse grid search, hardware acceleration, and math validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
from .optimized_labeler import EnhancedOptimizedTripleBarrierLabeler, CoarseGridConfig, HardwareOptimizationConfig
from .regime_aware import RegimeAwareTripleBarrierLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessor
from .cross_validation import LabelCrossValidator
from .utils import LabelingUtils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_market_data(num_rows: int = 10000) -> pd.DataFrame:
    """Generate realistic market data with multiple regimes."""
    logger.info(f"📊 Generating {num_rows} rows of realistic market data")
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='1min')
    
    # Create regime-based price series
    base_price = 100.0
    prices = [base_price]
    regimes = []
    
    for i in range(1, num_rows):
        # Define regime changes every 2000 periods
        regime_period = i // 2000
        regime = regime_period % 4
        
        if regime == 0:  # Bull market - strong upward trend
            drift = 0.0002
            volatility = 0.008
            regime_name = 'bull'
        elif regime == 1:  # Bear market - downward trend
            drift = -0.00015
            volatility = 0.012
            regime_name = 'bear'
        elif regime == 2:  # Sideways market - low volatility
            drift = 0.00005
            volatility = 0.004
            regime_name = 'sideways'
        else:  # Volatile market - high volatility, no clear trend
            drift = 0.0001
            volatility = 0.015
            regime_name = 'volatile'
        
        # Generate price change with regime-specific characteristics
        price_change = np.random.normal(drift, volatility)
        
        # Add some momentum persistence
        if i > 1:
            momentum = (prices[-1] - prices[-2]) / prices[-2] * 0.1
            price_change += momentum
        
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        regimes.append(regime_name)
    
    # Create OHLC data with realistic spreads
    data = pd.DataFrame({
        'open': prices,
        'close': prices,
        'volume': np.random.uniform(1000, 50000, num_rows)
    }, index=dates)
    
    # Generate realistic high/low prices
    spreads = np.random.uniform(0.0001, 0.001, num_rows)  # 0.01% to 0.1% spread
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + spreads)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - spreads)
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    logger.info(f"✅ Generated realistic market data: {data.shape}")
    return data, regimes

def run_enhanced_optimization_example():
    """Run the enhanced optimization example."""
    logger.info("🚀 Starting Enhanced Triple Barrier Labeling Example")
    
    # 1. Generate realistic market data
    market_data, regime_labels = generate_realistic_market_data(10000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    logger.info(f"📊 Regime distribution: {pd.Series(regime_labels).value_counts().to_dict()}")
    
    # 2. Configure enhanced optimization
    logger.info("\n--- Configuring Enhanced Optimization ---")
    
    # Coarse grid configuration
    coarse_config = CoarseGridConfig(
        pt_mult_range=(0.0005, 0.02),
        sl_mult_range=(0.0005, 0.01),
        time_barrier_range=(15, 120),
        lookahead_range=(50, 300),
        grid_size=15,  # Reduced for demo
        top_k_candidates=3
    )
    
    # Hardware optimization configuration
    hardware_config = HardwareOptimizationConfig(
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        enable_cpu_optimization=True,
        batch_size=2000,
        parallel_workers=4,
        use_vectorized_operations=True,
        enable_chunked_processing=True
    )
    
    # 3. Initialize enhanced labeler
    logger.info("\n--- Initializing Enhanced Labeler ---")
    enhanced_labeler = EnhancedOptimizedTripleBarrierLabeler({
        'coarse_grid_config': coarse_config,
        'hardware_config': hardware_config
    })
    
    # 4. Run coarse grid search
    logger.info("\n--- Running Coarse Grid Search ---")
    start_time = time.time()
    
    coarse_results = {}
    unique_regimes = regime_data['regime'].unique()
    
    for regime in unique_regimes:
        logger.info(f"🔍 Coarse grid search for regime: {regime}")
        regime_mask = regime_data['regime'] == regime
        regime_subset = market_data[regime_mask].copy()
        
        if len(regime_subset) < 100:
            logger.warning(f"⚠️ Insufficient data for regime {regime}")
            continue
        
        coarse_candidates = enhanced_labeler._coarse_grid_search(regime_subset, regime)
        coarse_results[regime] = coarse_candidates
        
        logger.info(f"✅ Found {len(coarse_candidates)} candidates for {regime}")
        if coarse_candidates:
            best = coarse_candidates[0]
            logger.info(f"   Best score: {best['score']:.4f}")
            logger.info(f"   Best params: PT={best['pt_mult']:.4f}, SL={best['sl_mult']:.4f}")
    
    coarse_time = time.time() - start_time
    logger.info(f"⏱️ Coarse grid search completed in {coarse_time:.2f}s")
    
    # 5. Run Bayesian optimization on top candidates
    logger.info("\n--- Running Bayesian Optimization ---")
    bayesian_start = time.time()
    
    optimization_results = enhanced_labeler.optimize_regime_parameters(
        data=market_data,
        regime_data=regime_data,
        n_trials=50  # Reduced for demo
    )
    
    bayesian_time = time.time() - bayesian_start
    logger.info(f"⏱️ Bayesian optimization completed in {bayesian_time:.2f}s")
    
    # 6. Print comprehensive report
    logger.info("\n--- Optimization Results ---")
    enhanced_labeler.print_optimization_report()
    
    # 7. Performance comparison
    logger.info("\n--- Performance Comparison ---")
    total_time = coarse_time + bayesian_time
    logger.info(f"📊 Total optimization time: {total_time:.2f}s")
    logger.info(f"   Coarse grid: {coarse_time:.2f}s ({coarse_time/total_time*100:.1f}%)")
    logger.info(f"   Bayesian: {bayesian_time:.2f}s ({bayesian_time/total_time*100:.1f}%)")
    
    # 8. Create optimized labels
    logger.info("\n--- Creating Optimized Labels ---")
    optimized_labels = enhanced_labeler.create_optimized_labels(
        data=market_data,
        regime_data=regime_data
    )
    
    logger.info(f"✅ Generated optimized labels: {optimized_labels.shape}")
    
    # 9. Quality assessment
    logger.info("\n--- Quality Assessment ---")
    quality_assessor = LabelQualityAssessor()
    quality_metrics = quality_assessor.assess_quality(
        labels_df=optimized_labels,
        original_data=market_data,
        regime_column='regime'
    )
    
    logger.info(f"📈 Quality metrics:")
    logger.info(f"   Overall quality: {quality_metrics.overall_quality:.3f}")
    logger.info(f"   Label distribution: {quality_metrics.label_distribution}")
    logger.info(f"   Regime balance: {quality_metrics.regime_balance}")
    
    return {
        'enhanced_labeler': enhanced_labeler,
        'optimized_labels': optimized_labels,
        'optimization_results': optimization_results,
        'coarse_results': coarse_results,
        'quality_metrics': quality_metrics,
        'timing': {
            'coarse_time': coarse_time,
            'bayesian_time': bayesian_time,
            'total_time': total_time
        }
    }

def run_hardware_comparison():
    """Compare performance with and without hardware optimizations."""
    logger.info("\n🔄 Running Hardware Optimization Comparison")
    
    # Generate smaller dataset for comparison
    market_data, regime_labels = generate_realistic_market_data(5000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    # Test with hardware optimizations
    logger.info("\n--- With Hardware Optimizations ---")
    hardware_config = HardwareOptimizationConfig(
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        enable_cpu_optimization=True,
        parallel_workers=4,
        use_vectorized_operations=True
    )
    
    start_time = time.time()
    enhanced_labeler = EnhancedOptimizedTripleBarrierLabeler({
        'hardware_config': hardware_config
    })
    
    # Run optimization
    enhanced_labeler.optimize_regime_parameters(market_data, regime_data, n_trials=20)
    hardware_time = time.time() - start_time
    
    # Test without hardware optimizations
    logger.info("\n--- Without Hardware Optimizations ---")
    no_hardware_config = HardwareOptimizationConfig(
        enable_gpu_acceleration=False,
        enable_memory_optimization=False,
        enable_cpu_optimization=False,
        parallel_workers=1,
        use_vectorized_operations=False
    )
    
    start_time = time.time()
    basic_labeler = EnhancedOptimizedTripleBarrierLabeler({
        'hardware_config': no_hardware_config
    })
    
    # Run optimization
    basic_labeler.optimize_regime_parameters(market_data, regime_data, n_trials=20)
    basic_time = time.time() - start_time
    
    # Compare results
    speedup = basic_time / hardware_time if hardware_time > 0 else 1.0
    
    logger.info(f"\n📊 Hardware Optimization Results:")
    logger.info(f"   With hardware: {hardware_time:.2f}s")
    logger.info(f"   Without hardware: {basic_time:.2f}s")
    logger.info(f"   Speedup: {speedup:.2f}x")
    
    return {
        'hardware_time': hardware_time,
        'basic_time': basic_time,
        'speedup': speedup
    }

def run_matrix_operations_demo():
    """Demonstrate matrix operations capabilities."""
    logger.info("\n🔧 Running Matrix Operations Demo")
    
    # Generate test data
    market_data, regime_labels = generate_realistic_market_data(2000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    # Test different matrix operation configurations
    configs = [
        ("Vectorized", HardwareOptimizationConfig(use_vectorized_operations=True)),
        ("Standard", HardwareOptimizationConfig(use_vectorized_operations=False)),
        ("Chunked", HardwareOptimizationConfig(enable_chunked_processing=True)),
        ("Batch", HardwareOptimizationConfig(batch_size=500))
    ]
    
    results = {}
    
    for config_name, config in configs:
        logger.info(f"\n--- Testing {config_name} Operations ---")
        
        start_time = time.time()
        labeler = EnhancedOptimizedTripleBarrierLabeler({
            'hardware_config': config
        })
        
        # Run a small optimization
        labeler.optimize_regime_parameters(market_data, regime_data, n_trials=10)
        elapsed_time = time.time() - start_time
        
        results[config_name] = elapsed_time
        logger.info(f"   {config_name}: {elapsed_time:.2f}s")
    
    # Find fastest configuration
    fastest = min(results, key=results.get)
    logger.info(f"\n🏆 Fastest configuration: {fastest} ({results[fastest]:.2f}s)")
    
    return results

if __name__ == "__main__":
    # Run the main enhanced example
    logger.info("🚀 Starting Enhanced Triple Barrier Labeling Demo")
    
    # Main example
    main_results = run_enhanced_optimization_example()
    
    # Hardware comparison
    hardware_results = run_hardware_comparison()
    
    # Matrix operations demo
    matrix_results = run_matrix_operations_demo()
    
    logger.info("\n🎉 Enhanced Triple Barrier Labeling Demo completed successfully!")
    
    # Summary
    logger.info("\n📊 SUMMARY")
    logger.info(f"   Total optimization time: {main_results['timing']['total_time']:.2f}s")
    logger.info(f"   Hardware speedup: {hardware_results['speedup']:.2f}x")
    logger.info(f"   Fastest matrix config: {min(matrix_results, key=matrix_results.get)}")
    logger.info(f"   Overall quality: {main_results['quality_metrics'].overall_quality:.3f}")