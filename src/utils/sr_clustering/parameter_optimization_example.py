"""
Example: Parameter Optimization for SR Level Detection

This example demonstrates how the parameter optimization system works to find the best
parameters for SR level detection, focusing on:
- Volume thresholds for SR confirmation
- Minimum touches required
- Bounce strength requirements
- Touch tolerance levels
- Quality scoring weights

The goal is to optimize these parameters to best identify high-quality SR levels.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_backtest_results(n_samples: int) -> List[Any]:
    """Create sample backtest results for testing."""
    from src.utils.sr_clustering.sr_backtesting_engine import BacktestResult, SRLevel
    
    results = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_samples):
        # Create a mock SR level
        level = SRLevel(
            price=100.0 + np.random.normal(0, 5),
            level_type='support' if i % 2 == 0 else 'resistance',
            strength=0.5 + np.random.random() * 0.5,
            detection_time=pd.Timestamp.now(),
            touches=2 + np.random.randint(0, 5)
        )
        
        # Create backtest result with realistic metrics
        success_rate = 0.3 + np.random.random() * 0.6  # 30-90% success rate
        avg_bounce_strength = 0.001 + np.random.random() * 0.01  # 0.1-1.1% bounce
        total_touches = 2 + np.random.randint(0, 8)
        time_persistence = 0.1 + np.random.random() * 0.9
        total_volume_at_level = 1000 + np.random.randint(0, 9000)
        avg_hold_time = 1 + np.random.randint(0, 24)
        
        # Calculate quality score based on these metrics
        quality_score = (
            success_rate * 0.3 +
            min(avg_bounce_strength * 100, 1.0) * 0.25 +
            min(total_touches / 10, 1.0) * 0.2 +
            time_persistence * 0.15 +
            min(total_volume_at_level / 10000, 1.0) * 0.1
        )
        
        result = BacktestResult(
            level=level,
            total_touches=total_touches,
            successful_touches=int(total_touches * success_rate),
            failed_touches=int(total_touches * (1 - success_rate)),
            success_rate=success_rate,
            avg_bounce_strength=avg_bounce_strength,
            max_bounce_strength=avg_bounce_strength * (1 + np.random.random()),
            avg_hold_time=avg_hold_time,
            total_volume_at_level=total_volume_at_level,
            price_deviation=0.01 + np.random.random() * 0.05,
            time_persistence=time_persistence,
            quality_score=quality_score
        )
        
        results.append(result)
    
    return results

def create_sample_market_data(n_samples: int) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
        'high': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)) + np.random.random(n_samples) * 2,
        'low': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)) - np.random.random(n_samples) * 2,
        'close': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
        'volume': 1000 + np.random.randint(0, 9000, n_samples)
    })
    
    return data

def test_parameter_optimization():
    """Test parameter optimization with different sample sizes."""
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    
    # Test different sample sizes
    sample_sizes = [10, 20, 30, 50, 100]
    
    for n_samples in sample_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing parameter optimization with {n_samples} samples")
        logger.info(f"{'='*60}")
        
        # Create sample data
        results = create_sample_backtest_results(n_samples)
        market_data = create_sample_market_data(n_samples)
        
        # Configure parameter optimization
        config = BacktestConfig(
            enable_parameter_optimization=True,
            parameter_optimization_method='grid_search',  # or 'genetic', 'scipy'
            min_samples_for_optimization=10,
            # Data-driven threshold calculation will be enabled automatically
        )
        
        # Create backtesting engine
        engine = SRBacktestingEngine(config)
        
        # Optimize parameters
        try:
            optimization_result = engine.optimize_sr_parameters(results, market_data=market_data)
            
            logger.info(f"✅ Parameter optimization completed for {n_samples} samples")
            logger.info(f"Optimization success: {optimization_result.get('optimization_success', False)}")
            logger.info(f"Optimization method: {optimization_result.get('optimization_method', 'unknown')}")
            logger.info(f"Optimization score: {optimization_result.get('optimization_score', 0.0):.4f}")
            
            # Show optimized parameters
            optimized_params = optimization_result.get('optimized_parameters', {})
            if optimized_params:
                logger.info("Optimized parameters:")
                logger.info(f"  - Touch tolerance: {optimized_params.get('touch_tolerance', 0):.4f}")
                logger.info(f"  - Min bounce strength: {optimized_params.get('min_bounce_strength', 0):.4f}")
                logger.info(f"  - Volume threshold multiplier: {optimized_params.get('volume_threshold_multiplier', 0):.2f}")
                logger.info(f"  - Min touches required: {optimized_params.get('min_touches_required', 0)}")
                logger.info(f"  - Max hold time: {optimized_params.get('max_hold_time', 0)} hours")
                
                logger.info("Quality scoring weights:")
                logger.info(f"  - Success rate weight: {optimized_params.get('success_rate_weight', 0):.3f}")
                logger.info(f"  - Bounce strength weight: {optimized_params.get('bounce_strength_weight', 0):.3f}")
                logger.info(f"  - Volume confirmation weight: {optimized_params.get('volume_confirmation_weight', 0):.3f}")
                logger.info(f"  - Time persistence weight: {optimized_params.get('time_persistence_weight', 0):.3f}")
                logger.info(f"  - Touch frequency weight: {optimized_params.get('touch_frequency_weight', 0):.3f}")
            
            # Show quality thresholds
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            if quality_thresholds:
                logger.info("Quality thresholds:")
                logger.info(f"  - Excellent: {quality_thresholds.get('excellent', 0):.3f}")
                logger.info(f"  - Good: {quality_thresholds.get('good', 0):.3f}")
                logger.info(f"  - Average: {quality_thresholds.get('average', 0):.3f}")
                logger.info(f"  - Poor: {quality_thresholds.get('poor', 0):.3f}")
            
        except Exception as e:
            logger.error(f"❌ Parameter optimization failed for {n_samples} samples: {e}")

def test_parameter_optimization_engine():
    """Test the parameter optimization engine directly."""
    from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing Parameter Optimization Engine Directly")
    logger.info(f"{'='*60}")
    
    # Test different sample sizes
    sample_sizes = [10, 20, 30, 50, 100]
    
    for n_samples in sample_sizes:
        logger.info(f"\n--- Testing with {n_samples} samples ---")
        
        # Create sample data
        results = create_sample_backtest_results(n_samples)
        market_data = create_sample_market_data(n_samples)
        
        # Configure parameter optimization
        config = ParameterOptimizationConfig(
            optimization_method='grid_search',  # or 'genetic', 'scipy'
            min_samples_for_optimization=10,
            adaptive_optimization=True,
            objective_metric='quality_score_correlation'  # or 'success_rate', 'composite'
        )
        
        # Use parameter optimization engine
        optimizer = get_parameter_optimization_engine(config)
        result = optimizer.optimize_parameters(results, market_data)
        
        logger.info(f"Optimization success: {result.optimization_success}")
        logger.info(f"Optimization method: {result.optimization_method}")
        logger.info(f"Best score: {result.best_score:.4f}")
        logger.info(f"Number of trials: {result.n_trials}")
        
        # Show best parameters
        best_params = result.best_parameters
        if best_params:
            logger.info("Best parameters:")
            for param, value in best_params.items():
                if isinstance(value, float):
                    logger.info(f"  {param}: {value:.4f}")
                else:
                    logger.info(f"  {param}: {value}")

def test_data_driven_thresholds():
    """Test data-driven threshold calculation."""
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing Data-Driven Threshold Calculation")
    logger.info(f"{'='*60}")
    
    # Create sample market data
    market_data = create_sample_market_data(1000)  # Large dataset for threshold calculation
    
    # Create backtesting engine
    config = BacktestConfig()
    engine = SRBacktestingEngine(config)
    
    # Calculate data-driven thresholds
    try:
        thresholds = engine.calculate_data_driven_thresholds(market_data)
        
        logger.info("Data-driven thresholds calculated:")
        logger.info(f"  - Touch tolerance: {thresholds.get('touch_tolerance', 0):.4f} ({thresholds.get('touch_tolerance', 0)*100:.2f}%)")
        logger.info(f"  - Min bounce strength: {thresholds.get('min_bounce_strength', 0):.4f} ({thresholds.get('min_bounce_strength', 0)*100:.2f}%)")
        logger.info(f"  - Max hold time: {thresholds.get('max_hold_time', 0)} hours")
        logger.info(f"  - Volume threshold multiplier: {thresholds.get('volume_threshold_multiplier', 0):.2f}")
        
    except Exception as e:
        logger.error(f"❌ Data-driven threshold calculation failed: {e}")

def demonstrate_parameter_impact():
    """Demonstrate how different parameters affect SR level detection."""
    logger.info(f"\n{'='*60}")
    logger.info("Demonstrating Parameter Impact on SR Level Detection")
    logger.info(f"{'='*60}")
    
    # Create sample data
    results = create_sample_backtest_results(50)
    market_data = create_sample_market_data(50)
    
    # Test different parameter configurations
    parameter_configs = [
        {
            'name': 'Conservative (High Volume, Many Touches)',
            'params': {
                'volume_threshold_multiplier': 2.5,
                'min_touches_required': 5,
                'touch_tolerance': 0.001,
                'min_bounce_strength': 0.002
            }
        },
        {
            'name': 'Moderate (Balanced)',
            'params': {
                'volume_threshold_multiplier': 1.5,
                'min_touches_required': 3,
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001
            }
        },
        {
            'name': 'Aggressive (Low Volume, Few Touches)',
            'params': {
                'volume_threshold_multiplier': 1.2,
                'min_touches_required': 2,
                'touch_tolerance': 0.005,
                'min_bounce_strength': 0.0005
            }
        }
    ]
    
    for config in parameter_configs:
        logger.info(f"\n--- {config['name']} ---")
        
        # Count how many results would pass each filter
        volume_threshold = config['params']['volume_threshold_multiplier'] * 1000  # Assume 1000 is avg volume
        min_touches = config['params']['min_touches_required']
        
        passed_volume = sum(1 for r in results if r.total_volume_at_level >= volume_threshold)
        passed_touches = sum(1 for r in results if r.total_touches >= min_touches)
        passed_both = sum(1 for r in results if r.total_volume_at_level >= volume_threshold and r.total_touches >= min_touches)
        
        logger.info(f"Results passing volume filter (>{volume_threshold}): {passed_volume}/{len(results)} ({passed_volume/len(results)*100:.1f}%)")
        logger.info(f"Results passing touch filter (>={min_touches}): {passed_touches}/{len(results)} ({passed_touches/len(results)*100:.1f}%)")
        logger.info(f"Results passing both filters: {passed_both}/{len(results)} ({passed_both/len(results)*100:.1f}%)")
        
        # Calculate average quality of filtered results
        if passed_both > 0:
            filtered_results = [r for r in results if r.total_volume_at_level >= volume_threshold and r.total_touches >= min_touches]
            avg_quality = np.mean([r.quality_score for r in filtered_results])
            avg_success_rate = np.mean([r.success_rate for r in filtered_results])
            logger.info(f"Average quality of filtered results: {avg_quality:.3f}")
            logger.info(f"Average success rate of filtered results: {avg_success_rate:.3f}")

if __name__ == "__main__":
    logger.info("Starting Parameter Optimization Tests")
    
    # Test parameter optimization
    test_parameter_optimization()
    
    # Test parameter optimization engine directly
    test_parameter_optimization_engine()
    
    # Test data-driven thresholds
    test_data_driven_thresholds()
    
    # Demonstrate parameter impact
    demonstrate_parameter_impact()
    
    logger.info("\n✅ All parameter optimization tests completed!")