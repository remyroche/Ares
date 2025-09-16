"""
Three-Stage Triple Barrier Labeling Example

This example demonstrates the enhanced three-stage optimization process:
1. Coarse Grid Search - Find promising parameter regions
2. Fine Grid Search - Refine around best coarse candidates  
3. Bayesian Optimization - Fine-tune with Optuna
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
from .optimized_labeler import (
    EnhancedOptimizedTripleBarrierLabeler, 
    CoarseGridConfig, 
    FineGridConfig,
    BayesianConfig,
    HardwareOptimizationConfig
)
from .regime_aware import RegimeAwareTripleBarrierLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessor
from .cross_validation import LabelCrossValidator
from .utils import LabelingUtils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_complex_market_data(num_rows: int = 15000) -> pd.DataFrame:
    """Generate complex market data with multiple regimes and realistic patterns."""
    logger.info(f"📊 Generating {num_rows} rows of complex market data")
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='1min')
    
    # Create regime-based price series with more complexity
    base_price = 100.0
    prices = [base_price]
    regimes = []
    
    for i in range(1, num_rows):
        # Define regime changes with varying lengths
        regime_period = i // 2500  # Change every 2500 periods
        regime = regime_period % 5
        
        if regime == 0:  # Strong bull market
            drift = 0.0003
            volatility = 0.006
            regime_name = 'strong_bull'
        elif regime == 1:  # Moderate bull market
            drift = 0.0001
            volatility = 0.008
            regime_name = 'moderate_bull'
        elif regime == 2:  # Bear market
            drift = -0.0002
            volatility = 0.012
            regime_name = 'bear'
        elif regime == 3:  # Sideways market
            drift = 0.00005
            volatility = 0.004
            regime_name = 'sideways'
        else:  # High volatility market
            drift = 0.0001
            volatility = 0.018
            regime_name = 'high_vol'
        
        # Generate price change with regime-specific characteristics
        price_change = np.random.normal(drift, volatility)
        
        # Add momentum and mean reversion effects
        if i > 10:
            # Short-term momentum
            recent_returns = np.array(prices[-10:]) / np.array(prices[-11:-1]) - 1
            momentum = np.mean(recent_returns) * 0.1
            price_change += momentum
            
            # Mean reversion for extreme moves
            if abs(price_change) > 2 * volatility:
                price_change *= 0.5
        
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        regimes.append(regime_name)
    
    # Create OHLC data with realistic spreads and gaps
    data = pd.DataFrame({
        'open': prices,
        'close': prices,
        'volume': np.random.uniform(1000, 50000, num_rows)
    }, index=dates)
    
    # Generate realistic high/low prices with intraday volatility
    intraday_vol = np.random.uniform(0.0005, 0.002, num_rows)
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + intraday_vol)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - intraday_vol)
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    logger.info(f"✅ Generated complex market data: {data.shape}")
    return data, regimes

def run_three_stage_optimization_example():
    """Run the three-stage optimization example."""
    logger.info("🚀 Starting Three-Stage Triple Barrier Labeling Example")
    
    # 1. Generate complex market data
    market_data, regime_labels = generate_complex_market_data(15000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    logger.info(f"📊 Regime distribution: {pd.Series(regime_labels).value_counts().to_dict()}")
    
    # 2. Configure three-stage optimization
    logger.info("\n--- Configuring Three-Stage Optimization ---")
    
    # Coarse grid configuration (first stage)
    coarse_config = CoarseGridConfig(
        pt_mult_range=(0.0005, 0.025),
        sl_mult_range=(0.0005, 0.015),
        time_barrier_range=(10, 150),
        lookahead_range=(30, 400),
        grid_size=12,  # 12³ = 1,728 combinations
        top_k_candidates=6
    )
    
    # Fine grid configuration (second stage)
    fine_config = FineGridConfig(
        refinement_factor=0.25,  # Narrow to 25% of original range
        grid_size=8,  # 8³ = 512 combinations
        top_k_candidates=3,
        min_range_size=0.0005
    )
    
    # Bayesian configuration (third stage)
    bayesian_config = BayesianConfig(
        n_trials=75,  # Reduced for demo
        timeout=300,  # 5 minutes timeout
        early_stopping_patience=15,
        objective_function="combined",  # Use combined objective
        acquisition_function="EI",
        random_state=42
    )
    
    # Hardware optimization configuration
    hardware_config = HardwareOptimizationConfig(
        enable_gpu_acceleration=True,
        enable_memory_optimization=True,
        enable_cpu_optimization=True,
        batch_size=3000,
        parallel_workers=6,
        use_vectorized_operations=True,
        enable_chunked_processing=True
    )
    
    # 3. Initialize enhanced labeler with three-stage configuration
    logger.info("\n--- Initializing Three-Stage Labeler ---")
    three_stage_labeler = EnhancedOptimizedTripleBarrierLabeler({
        'coarse_grid_config': coarse_config,
        'fine_grid_config': fine_config,
        'bayesian_config': bayesian_config,
        'hardware_config': hardware_config
    })
    
    # 4. Run three-stage optimization
    logger.info("\n--- Running Three-Stage Optimization ---")
    start_time = time.time()
    
    optimization_results = three_stage_labeler.optimize_regime_parameters(
        data=market_data,
        regime_data=regime_data
    )
    
    total_time = time.time() - start_time
    logger.info(f"⏱️ Total optimization time: {total_time:.2f}s")
    
    # 5. Print detailed three-stage report
    logger.info("\n--- Three-Stage Optimization Results ---")
    three_stage_labeler.print_optimization_report()
    
    # 6. Analyze stage performance
    logger.info("\n--- Stage Performance Analysis ---")
    stage_times = {
        'Coarse Grid': optimization_results.get('coarse_time', 0),
        'Fine Grid': optimization_results.get('fine_time', 0),
        'Bayesian': optimization_results.get('bayesian_time', 0)
    }
    
    total_opt_time = sum(stage_times.values())
    for stage, time_taken in stage_times.items():
        percentage = (time_taken / total_opt_time * 100) if total_opt_time > 0 else 0
        logger.info(f"   {stage}: {time_taken:.2f}s ({percentage:.1f}%)")
    
    # 7. Create optimized labels
    logger.info("\n--- Creating Optimized Labels ---")
    optimized_labels = three_stage_labeler.create_optimized_labels(
        data=market_data,
        regime_data=regime_data
    )
    
    logger.info(f"✅ Generated optimized labels: {optimized_labels.shape}")
    
    # 8. Quality assessment
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
        'three_stage_labeler': three_stage_labeler,
        'optimized_labels': optimized_labels,
        'optimization_results': optimization_results,
        'quality_metrics': quality_metrics,
        'stage_times': stage_times,
        'total_time': total_time
    }

def run_optimization_comparison():
    """Compare different optimization approaches."""
    logger.info("\n🔄 Running Optimization Approach Comparison")
    
    # Generate smaller dataset for comparison
    market_data, regime_labels = generate_complex_market_data(8000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    # Test different configurations
    configs = [
        ("Coarse Only", {
            'coarse_grid_config': CoarseGridConfig(grid_size=15, top_k_candidates=1),
            'fine_grid_config': FineGridConfig(grid_size=1, top_k_candidates=1),
            'bayesian_config': BayesianConfig(n_trials=1)
        }),
        ("Coarse + Fine", {
            'coarse_grid_config': CoarseGridConfig(grid_size=12, top_k_candidates=3),
            'fine_grid_config': FineGridConfig(grid_size=8, top_k_candidates=1),
            'bayesian_config': BayesianConfig(n_trials=1)
        }),
        ("Three-Stage", {
            'coarse_grid_config': CoarseGridConfig(grid_size=10, top_k_candidates=4),
            'fine_grid_config': FineGridConfig(grid_size=6, top_k_candidates=2),
            'bayesian_config': BayesianConfig(n_trials=30)
        })
    ]
    
    results = {}
    
    for config_name, config in configs:
        logger.info(f"\n--- Testing {config_name} ---")
        
        start_time = time.time()
        labeler = EnhancedOptimizedTripleBarrierLabeler(config)
        
        # Run optimization
        opt_results = labeler.optimize_regime_parameters(market_data, regime_data)
        elapsed_time = time.time() - start_time
        
        # Calculate average optimization score
        avg_score = 0
        if opt_results.get('regime_parameters'):
            scores = [params.get('optimization_score', 0) for params in opt_results['regime_parameters'].values()]
            avg_score = np.mean(scores) if scores else 0
        
        results[config_name] = {
            'time': elapsed_time,
            'avg_score': avg_score,
            'regimes': len(opt_results.get('regime_parameters', {}))
        }
        
        logger.info(f"   Time: {elapsed_time:.2f}s")
        logger.info(f"   Avg Score: {avg_score:.4f}")
        logger.info(f"   Regimes: {results[config_name]['regimes']}")
    
    # Find best approach
    best_time = min(results, key=lambda x: results[x]['time'])
    best_score = max(results, key=lambda x: results[x]['avg_score'])
    
    logger.info(f"\n🏆 Best Results:")
    logger.info(f"   Fastest: {best_time} ({results[best_time]['time']:.2f}s)")
    logger.info(f"   Highest Score: {best_score} ({results[best_score]['avg_score']:.4f})")
    
    return results

def run_parameter_sensitivity_analysis():
    """Analyze sensitivity of optimization to parameter settings."""
    logger.info("\n🔬 Running Parameter Sensitivity Analysis")
    
    # Generate test data
    market_data, regime_labels = generate_complex_market_data(6000)
    regime_data = pd.DataFrame({'regime': regime_labels}, index=market_data.index)
    
    # Test different refinement factors
    refinement_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    for factor in refinement_factors:
        logger.info(f"\n--- Testing Refinement Factor: {factor} ---")
        
        config = {
            'coarse_grid_config': CoarseGridConfig(grid_size=8, top_k_candidates=3),
            'fine_grid_config': FineGridConfig(refinement_factor=factor, grid_size=6, top_k_candidates=2),
            'bayesian_config': BayesianConfig(n_trials=20)
        }
        
        start_time = time.time()
        labeler = EnhancedOptimizedTripleBarrierLabeler(config)
        
        opt_results = labeler.optimize_regime_parameters(market_data, regime_data)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        avg_score = 0
        if opt_results.get('regime_parameters'):
            scores = [params.get('optimization_score', 0) for params in opt_results['regime_parameters'].values()]
            avg_score = np.mean(scores) if scores else 0
        
        results[factor] = {
            'time': elapsed_time,
            'avg_score': avg_score,
            'coarse_time': opt_results.get('coarse_time', 0),
            'fine_time': opt_results.get('fine_time', 0),
            'bayesian_time': opt_results.get('bayesian_time', 0)
        }
        
        logger.info(f"   Total Time: {elapsed_time:.2f}s")
        logger.info(f"   Avg Score: {avg_score:.4f}")
        logger.info(f"   Stage Times: C={results[factor]['coarse_time']:.1f}s, F={results[factor]['fine_time']:.1f}s, B={results[factor]['bayesian_time']:.1f}s")
    
    # Find optimal refinement factor
    best_factor = max(results, key=lambda x: results[x]['avg_score'])
    logger.info(f"\n🎯 Optimal refinement factor: {best_factor} (score: {results[best_factor]['avg_score']:.4f})")
    
    return results

if __name__ == "__main__":
    # Run the main three-stage example
    logger.info("🚀 Starting Three-Stage Triple Barrier Labeling Demo")
    
    # Main example
    main_results = run_three_stage_optimization_example()
    
    # Optimization comparison
    comparison_results = run_optimization_comparison()
    
    # Parameter sensitivity analysis
    sensitivity_results = run_parameter_sensitivity_analysis()
    
    logger.info("\n🎉 Three-Stage Triple Barrier Labeling Demo completed successfully!")
    
    # Summary
    logger.info("\n📊 SUMMARY")
    logger.info(f"   Total optimization time: {main_results['total_time']:.2f}s")
    logger.info(f"   Stage breakdown: {main_results['stage_times']}")
    logger.info(f"   Overall quality: {main_results['quality_metrics'].overall_quality:.3f}")
    logger.info(f"   Best optimization approach: {max(comparison_results, key=lambda x: comparison_results[x]['avg_score'])}")
    logger.info(f"   Optimal refinement factor: {max(sensitivity_results, key=lambda x: sensitivity_results[x]['avg_score'])}")