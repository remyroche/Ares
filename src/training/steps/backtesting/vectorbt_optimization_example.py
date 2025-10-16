"""
VectorBT Optimization Example for Backtesting Parameter Optimization

This example demonstrates how to use the VectorBT optimizations
in the backtesting parameter optimization system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the optimized components
from final_parameters_optimization import FinalParametersOptimizer, EvaluationMetrics
from nas_tas.validation_orchestrator import ValidationOrchestrator, ValidationConfig, ValidationMode
from nas_tas.walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMode
from nas_tas.performance_attribution import PerformanceAttributor, AttributionConfig, AttributionMethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate price data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1min')
    returns = np.random.normal(0, 0.01, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)

    return data

def example_final_parameters_optimization():
    """Example of using VectorBT-optimized final parameters optimization."""
    print("🚀 Example: Final Parameters Optimization with VectorBT")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(5000)
    print(f"📊 Created sample data: {data.shape}")

    # Configuration for VectorBT optimization
    config = {
        'n_trials': 20,
        'timeout': 60,
        'enable_vectorbt_optimization': True,
        'enable_hardware_optimization': True,
        'enable_parallel_evaluation': True,
        'max_workers': 4,
        'chunk_size': 1000,
        'max_memory_gb': 8.0
    }

    # Initialize optimizer with VectorBT
    optimizer = FinalParametersOptimizer(config)

    # Add some parameters to optimize
    optimizer.add_parameter('confidence_threshold', 'float', (0.1, 0.9))
    optimizer.add_parameter('position_size', 'float', (0.01, 0.1))
    optimizer.add_parameter('stop_loss', 'float', (0.01, 0.05))

    # Define objective function
    def objective_function(params):
        """Simple objective function for demonstration."""
        # Simulate some computation
        returns = data['close'].pct_change().dropna()

        # Use VectorBT rolling operations if available
        if optimizer.vectorbt_enabled:
            volatility = optimizer.rolling_optimizer.rolling_std(returns, window=20)
            momentum = optimizer.rolling_optimizer.rolling_mean(returns, window=20)
        else:
            volatility = returns.rolling(window=20).std()
            momentum = returns.rolling(window=20).mean()

        # Calculate simple score
        sharpe_ratio = momentum.mean() / volatility.mean() if volatility.mean() > 0 else 0
        score = sharpe_ratio * params['confidence_threshold'] * params['position_size']

        return max(0, min(1, score))  # Normalize to [0, 1]

    # Run optimization
    print("🔧 Running parameter optimization...")
    results = optimizer.optimize_parameters(objective_function)

    print(f"✅ Optimization completed!")
    print(f"   Best parameters: {results['best_parameters']}")
    print(f"   Best score: {results['best_score']:.4f}")

    # Get VectorBT performance stats
    vectorbt_stats = optimizer.get_vectorbt_performance_stats()
    print(f"📈 VectorBT Performance Stats:")
    print(f"   VectorBT enabled: {vectorbt_stats.get('vectorbt_enabled', False)}")
    if vectorbt_stats.get('vectorbt_enabled'):
        print(f"   Rolling operations: {vectorbt_stats.get('rolling_operations', 0)}")
        print(f"   Batch operations: {vectorbt_stats.get('batch_operations', 0)}")
        print(f"   Total VectorBT time: {vectorbt_stats.get('total_vectorbt_time', 0):.3f}s")

def example_validation_orchestrator():
    """Example of using VectorBT-optimized validation orchestrator."""
    print("\n🚀 Example: Validation Orchestrator with VectorBT")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(3000)
    print(f"📊 Created sample data: {data.shape}")

    # Configuration
    config = ValidationConfig(
        mode=ValidationMode.COMPREHENSIVE,
        enable_backtesting=True,
        enable_walk_forward=True,
        enable_attribution=True,
        enable_scenario_testing=True,
        enable_gpu=False,
        enable_parallel=True,
        chunk_size=1000
    )

    # Initialize orchestrator
    orchestrator = ValidationOrchestrator(config)

    # Test feature engineering with VectorBT
    print("🔧 Testing feature engineering with VectorBT...")
    engineered_data = orchestrator._engineer_features(data)

    print(f"✅ Feature engineering completed!")
    print(f"   Original shape: {data.shape}")
    print(f"   Engineered shape: {engineered_data.shape}")
    print(f"   New features: {[col for col in engineered_data.columns if col not in data.columns]}")

    # Check if VectorBT was used
    if hasattr(orchestrator, 'rolling_optimizer') and orchestrator.rolling_optimizer is not None:
        rolling_stats = orchestrator.rolling_optimizer.get_performance_stats()
        print(f"📈 VectorBT Rolling Stats:")
        print(f"   Total operations: {rolling_stats.get('total_operations', 0)}")
        print(f"   VectorBT operations: {rolling_stats.get('vectorbt_operations', 0)}")
        print(f"   Average time per operation: {rolling_stats.get('avg_time_per_operation', 0):.4f}s")

def example_walk_forward_analysis():
    """Example of using VectorBT-optimized walk-forward analysis."""
    print("\n🚀 Example: Walk-Forward Analysis with VectorBT")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(2000)
    print(f"📊 Created sample data: {data.shape}")

    # Configuration
    config = WalkForwardConfig(
        mode=WalkForwardMode.ROLLING_WINDOW,
        initial_training_size=1000,
        validation_size=200,
        step_size=100,
        enable_purging=True,
        enable_leakage_detection=True,
        enable_gpu=False,
        enable_parallel=True,
        chunk_size=500
    )

    # Initialize analyzer
    analyzer = WalkForwardAnalyzer(config)

    # Test regime change detection with VectorBT
    print("🔧 Testing regime change detection with VectorBT...")
    regime_changes = analyzer._detect_regime_changes(data)

    print(f"✅ Regime change detection completed!")
    print(f"   Regime changes detected: {len(regime_changes)}")

    # Check VectorBT usage
    if hasattr(analyzer, 'rolling_optimizer') and analyzer.rolling_optimizer is not None:
        rolling_stats = analyzer.rolling_optimizer.get_performance_stats()
        print(f"📈 VectorBT Rolling Stats:")
        print(f"   Total operations: {rolling_stats.get('total_operations', 0)}")
        print(f"   VectorBT operations: {rolling_stats.get('vectorbt_operations', 0)}")

def example_performance_attribution():
    """Example of using VectorBT-optimized performance attribution."""
    print("\n🚀 Example: Performance Attribution with VectorBT")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(1500)
    print(f"📊 Created sample data: {data.shape}")

    # Configuration
    config = AttributionConfig(
        attribution_method=AttributionMethod.FACTOR_BASED,
        enable_regime_attribution=True,
        enable_model_attribution=True,
        enable_factor_attribution=True,
        enable_gpu=False,
        enable_parallel=True,
        chunk_size=500
    )

    # Initialize attributor
    attributor = PerformanceAttributor(config)

    # Test factor data calculation with VectorBT
    print("🔧 Testing factor data calculation with VectorBT...")
    factor_data = attributor._calculate_factor_data(data)

    print(f"✅ Factor data calculation completed!")
    print(f"   Factors calculated: {list(factor_data.keys())}")

    # Check VectorBT usage
    if hasattr(attributor, 'rolling_optimizer') and attributor.rolling_optimizer is not None:
        rolling_stats = attributor.rolling_optimizer.get_performance_stats()
        print(f"📈 VectorBT Rolling Stats:")
        print(f"   Total operations: {rolling_stats.get('total_operations', 0)}")
        print(f"   VectorBT operations: {rolling_stats.get('vectorbt_operations', 0)}")

def performance_comparison():
    """Compare performance with and without VectorBT optimization."""
    print("\n🚀 Performance Comparison: VectorBT vs Pandas")
    print("=" * 60)

    # Create larger dataset for comparison
    data = create_sample_data(10000)
    returns = data['close'].pct_change().dropna()

    print(f"📊 Testing with {len(returns)} data points")

    # Test pandas rolling operations
    import time

    print("🔧 Testing pandas rolling operations...")
    start_time = time.time()

    pandas_volatility = returns.rolling(window=20).std()
    pandas_momentum = returns.rolling(window=20).mean()
    pandas_skewness = returns.rolling(window=20).skew()

    pandas_time = time.time() - start_time
    print(f"   Pandas time: {pandas_time:.4f}s")

    # Test VectorBT rolling operations
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer

        print("🔧 Testing VectorBT rolling operations...")
        rolling_optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=False,
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000
        )

        start_time = time.time()

        vectorbt_volatility = rolling_optimizer.rolling_std(returns, window=20)
        vectorbt_momentum = rolling_optimizer.rolling_mean(returns, window=20)
        vectorbt_skewness = rolling_optimizer.rolling_skew(returns, window=20)

        vectorbt_time = time.time() - start_time
        print(f"   VectorBT time: {vectorbt_time:.4f}s")

        # Calculate speedup
        speedup = pandas_time / vectorbt_time if vectorbt_time > 0 else 0
        print(f"   Speedup: {speedup:.2f}x")

        # Get VectorBT stats
        stats = rolling_optimizer.get_performance_stats()
        print(f"📈 VectorBT Performance Stats:")
        print(f"   Total operations: {stats.get('total_operations', 0)}")
        print(f"   VectorBT operations: {stats.get('vectorbt_operations', 0)}")
        print(f"   Average time per operation: {stats.get('avg_time_per_operation', 0):.4f}s")

    except ImportError:
        print("   VectorBT not available for comparison")

def main():
    """Run all examples."""
    print("🚀 VectorBT Optimization Examples for Backtesting Parameter Optimization")
    print("=" * 80)

    try:
        # Run examples
        example_final_parameters_optimization()
        example_validation_orchestrator()
        example_walk_forward_analysis()
        example_performance_attribution()
        performance_comparison()

        print("\n✅ All examples completed successfully!")
        print("\n📋 Summary of VectorBT Optimizations Implemented:")
        print("   • VectorBTRollingOptimizer integration in FinalParametersOptimizer")
        print("   • UnifiedVectorizationManager for parameter evaluation")
        print("   • Optimized rolling operations in ValidationOrchestrator")
        print("   • VectorBT rolling calculations in WalkForwardAnalyzer")
        print("   • Enhanced performance metrics in PerformanceAttributor")
        print("   • Batch parameter evaluation with VectorBT optimization")
        print("   • Memory-efficient data processing")
        print("   • GPU acceleration support (when available)")
        print("   • Comprehensive performance monitoring and statistics")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        logger.exception("Example execution failed")

if __name__ == "__main__":
    main()
