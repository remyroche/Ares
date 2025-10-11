"""
VectorBT Integration Example

This module demonstrates how to use the VectorBT-enhanced components
for backtesting, financial metrics, and portfolio optimization.

Example usage:
    python vectorbt_integration_example.py
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

# Import VectorBT components
from .vectorbt_backtesting_engine import (
    VectorBTBacktestingEngine, VectorBTBacktestConfig, BacktestMode,
    run_vectorbt_backtest, create_vectorbt_config
)
from .vectorbt_financial_metrics import (
    VectorBTFinancialMetrics, FinancialMetricsConfig,
    calculate_financial_metrics, create_metrics_config
)
from .vectorbt_portfolio_optimization import (
    VectorBTPortfolioOptimizer, OptimizationConfig, OptimizationMethod,
    optimize_portfolio, create_optimization_config
)
from .unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy,
    optimize_vectorbt_backtesting, optimize_vectorbt_metrics, optimize_vectorbt_portfolio
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_periods: int = 1000, n_assets: int = 5, seed: int = 42):
    """Generate sample financial data for testing."""
    np.random.seed(seed)
    
    # Generate random returns
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
    
    # Generate prices from returns
    prices = 100 * (1 + returns).cumprod(axis=0)
    
    # Generate signals (simple momentum strategy)
    signals = np.zeros_like(returns)
    for i in range(20, n_periods):
        for j in range(n_assets):
            if returns[i-20:i, j].mean() > 0.001:
                signals[i, j] = 1  # Buy
            elif returns[i-20:i, j].mean() < -0.001:
                signals[i, j] = -1  # Sell
    
    # Generate timestamps
    timestamps = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')
    
    # Create asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    return {
        'returns': returns,
        'prices': prices,
        'signals': signals,
        'timestamps': timestamps,
        'asset_names': asset_names
    }


def demonstrate_vectorbt_backtesting():
    """Demonstrate VectorBT backtesting capabilities."""
    print("\n" + "="*60)
    print("🚀 VECTORBT BACKTESTING DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    data = generate_sample_data(n_periods=1000, n_assets=3)
    
    # Create VectorBT backtesting configuration
    config = create_vectorbt_config(
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        use_gpu=True
    )
    
    # Test different backtesting modes
    modes = [BacktestMode.VECTORBT_CPU, BacktestMode.VECTORBT_PARALLEL]
    
    results = {}
    
    for mode in modes:
        print(f"\n🔄 Testing {mode.value}...")
        
        try:
            start_time = time.time()
            result = run_vectorbt_backtest(
                data['signals'], 
                data['prices'], 
                data['timestamps'],
                config=config,
                mode=mode
            )
            execution_time = time.time() - start_time
            
            results[mode.value] = {
                'execution_time': execution_time,
                'final_value': result.portfolio_values[-1],
                'total_return': result.performance_metrics['total_return'],
                'sharpe_ratio': result.performance_metrics['sharpe_ratio'],
                'max_drawdown': result.performance_metrics['max_drawdown']
            }
            
            print(f"✅ {mode.value}:")
            print(f"   Execution time: {execution_time:.3f}s")
            print(f"   Final value: ${result.portfolio_values[-1]:,.2f}")
            print(f"   Total return: {result.performance_metrics['total_return']:.2%}")
            print(f"   Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.3f}")
            print(f"   Max drawdown: {result.performance_metrics['max_drawdown']:.2%}")
            
        except Exception as e:
            print(f"❌ {mode.value} failed: {e}")
            results[mode.value] = {'error': str(e)}
    
    return results


def demonstrate_vectorbt_metrics():
    """Demonstrate VectorBT financial metrics capabilities."""
    print("\n" + "="*60)
    print("📊 VECTORBT FINANCIAL METRICS DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    data = generate_sample_data(n_periods=1000, n_assets=3)
    
    # Calculate portfolio values from returns
    portfolio_values = 100000 * (1 + data['returns'].sum(axis=1)).cumprod()
    
    # Create benchmark data
    benchmark_returns = np.random.normal(0.0008, 0.015, len(data['returns']))
    benchmark_values = 100000 * (1 + benchmark_returns).cumprod()
    
    # Create VectorBT metrics configuration
    config = create_metrics_config(
        risk_free_rate=0.02,
        annualization_factor=252,
        enable_regime_analysis=True
    )
    
    print("\n🔄 Calculating comprehensive financial metrics...")
    
    try:
        start_time = time.time()
        metrics = calculate_financial_metrics(
            portfolio_values=portfolio_values,
            returns=data['returns'].sum(axis=1),
            benchmark_values=benchmark_values,
            timestamps=data['timestamps'],
            config=config
        )
        execution_time = time.time() - start_time
        
        print(f"✅ Metrics calculated in {execution_time:.3f}s")
        print(f"📊 Total metrics calculated: {len(metrics)}")
        
        # Display key metrics by category
        categories = {
            'Returns': ['total_return', 'annualized_return', 'cumulative_return'],
            'Risk': ['volatility', 'var_95', 'cvar_95', 'skewness', 'kurtosis'],
            'Risk-Adjusted': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
            'Drawdown': ['max_drawdown', 'avg_drawdown', 'recovery_time'],
            'Trading': ['win_rate', 'profit_factor', 'expectancy'],
            'Benchmark': ['alpha', 'beta', 'tracking_error', 'relative_performance']
        }
        
        for category, metric_names in categories.items():
            print(f"\n📈 {category} Metrics:")
            for metric_name in metric_names:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        if 'return' in metric_name or 'drawdown' in metric_name:
                            print(f"   {metric_name}: {value:.2%}")
                        elif 'ratio' in metric_name:
                            print(f"   {metric_name}: {value:.3f}")
                        else:
                            print(f"   {metric_name}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return {}


def demonstrate_vectorbt_portfolio_optimization():
    """Demonstrate VectorBT portfolio optimization capabilities."""
    print("\n" + "="*60)
    print("🎯 VECTORBT PORTFOLIO OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    data = generate_sample_data(n_periods=1000, n_assets=5)
    
    # Test different optimization methods
    methods = [
        OptimizationMethod.MEAN_VARIANCE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.MIN_VARIANCE,
        OptimizationMethod.MAX_SHARPE
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n🔄 Testing {method.value}...")
        
        try:
            start_time = time.time()
            result = optimize_portfolio(
                returns=data['returns'],
                method=method,
                asset_names=data['asset_names']
            )
            execution_time = time.time() - start_time
            
            results[method.value] = {
                'execution_time': execution_time,
                'weights': result.weights,
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'sharpe_ratio': result.sharpe_ratio
            }
            
            print(f"✅ {method.value}:")
            print(f"   Execution time: {execution_time:.3f}s")
            print(f"   Expected return: {result.expected_return:.2%}")
            print(f"   Expected volatility: {result.expected_volatility:.2%}")
            print(f"   Sharpe ratio: {result.sharpe_ratio:.3f}")
            print(f"   Weights: {result.weights}")
            
        except Exception as e:
            print(f"❌ {method.value} failed: {e}")
            results[method.value] = {'error': str(e)}
    
    return results


def demonstrate_unified_vectorization_manager():
    """Demonstrate unified vectorization manager with VectorBT."""
    print("\n" + "="*60)
    print("🔧 UNIFIED VECTORIZATION MANAGER DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    data = generate_sample_data(n_periods=500, n_assets=3)
    
    # Test VectorBT operations through unified manager
    print("\n🔄 Testing VectorBT backtesting through unified manager...")
    
    try:
        result = optimize_vectorbt_backtesting(
            signals=data['signals'],
            prices=data['prices'],
            timestamps=data['timestamps']
        )
        
        print(f"✅ VectorBT backtesting completed:")
        print(f"   Strategy used: {result.strategy_used}")
        print(f"   Computation time: {result.computation_time:.3f}s")
        print(f"   Performance gain: {result.performance_gain:.2f}x")
        
    except Exception as e:
        print(f"❌ VectorBT backtesting failed: {e}")
    
    print("\n🔄 Testing VectorBT metrics through unified manager...")
    
    try:
        portfolio_values = 100000 * (1 + data['returns'].sum(axis=1)).cumprod()
        
        result = optimize_vectorbt_metrics(
            portfolio_values=portfolio_values,
            returns=data['returns'].sum(axis=1),
            timestamps=data['timestamps']
        )
        
        print(f"✅ VectorBT metrics completed:")
        print(f"   Strategy used: {result.strategy_used}")
        print(f"   Computation time: {result.computation_time:.3f}s")
        print(f"   Performance gain: {result.performance_gain:.2f}x")
        
    except Exception as e:
        print(f"❌ VectorBT metrics failed: {e}")
    
    print("\n🔄 Testing VectorBT portfolio optimization through unified manager...")
    
    try:
        result = optimize_vectorbt_portfolio(
            returns=data['returns'],
            asset_names=data['asset_names']
        )
        
        print(f"✅ VectorBT portfolio optimization completed:")
        print(f"   Strategy used: {result.strategy_used}")
        print(f"   Computation time: {result.computation_time:.3f}s")
        print(f"   Performance gain: {result.performance_gain:.2f}x")
        
    except Exception as e:
        print(f"❌ VectorBT portfolio optimization failed: {e}")
    
    # Get optimization statistics
    manager = UnifiedVectorizationManager()
    stats = manager.get_optimization_stats()
    
    print(f"\n📊 Optimization Statistics:")
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   Average speedup: {stats['average_speedup']:.2f}x")
    print(f"   Available optimizations: {stats['available_optimizations']}")


def run_performance_benchmark():
    """Run performance benchmark comparing different approaches."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Generate larger dataset for benchmarking
    data = generate_sample_data(n_periods=5000, n_assets=5)
    
    print(f"\n📊 Benchmarking with {len(data['returns'])} periods and {data['returns'].shape[1]} assets...")
    
    # Benchmark VectorBT backtesting
    print("\n🔄 Benchmarking VectorBT backtesting...")
    
    config = create_vectorbt_config(initial_capital=100000.0)
    engine = VectorBTBacktestingEngine(config)
    
    start_time = time.time()
    result = engine.run_backtest(
        data['signals'], 
        data['prices'], 
        data['timestamps'],
        mode=BacktestMode.VECTORBT_CPU
    )
    vectorbt_time = time.time() - start_time
    
    print(f"✅ VectorBT backtesting: {vectorbt_time:.3f}s")
    print(f"   Final value: ${result.portfolio_values[-1]:,.2f}")
    print(f"   Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.3f}")
    
    # Benchmark VectorBT metrics
    print("\n🔄 Benchmarking VectorBT metrics...")
    
    portfolio_values = 100000 * (1 + data['returns'].sum(axis=1)).cumprod()
    
    start_time = time.time()
    metrics = calculate_financial_metrics(
        portfolio_values=portfolio_values,
        returns=data['returns'].sum(axis=1),
        timestamps=data['timestamps']
    )
    metrics_time = time.time() - start_time
    
    print(f"✅ VectorBT metrics: {metrics_time:.3f}s")
    print(f"   Metrics calculated: {len(metrics)}")
    print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    
    # Benchmark VectorBT portfolio optimization
    print("\n🔄 Benchmarking VectorBT portfolio optimization...")
    
    start_time = time.time()
    opt_result = optimize_portfolio(
        returns=data['returns'],
        method=OptimizationMethod.MEAN_VARIANCE,
        asset_names=data['asset_names']
    )
    optimization_time = time.time() - start_time
    
    print(f"✅ VectorBT portfolio optimization: {optimization_time:.3f}s")
    print(f"   Expected return: {opt_result.expected_return:.2%}")
    print(f"   Expected volatility: {opt_result.expected_volatility:.2%}")
    print(f"   Sharpe ratio: {opt_result.sharpe_ratio:.3f}")
    
    # Summary
    total_time = vectorbt_time + metrics_time + optimization_time
    print(f"\n📊 Benchmark Summary:")
    print(f"   Total execution time: {total_time:.3f}s")
    print(f"   Backtesting: {vectorbt_time:.3f}s ({vectorbt_time/total_time*100:.1f}%)")
    print(f"   Metrics: {metrics_time:.3f}s ({metrics_time/total_time*100:.1f}%)")
    print(f"   Optimization: {optimization_time:.3f}s ({optimization_time/total_time*100:.1f}%)")


def main():
    """Main demonstration function."""
    print("🚀 VectorBT Integration Demonstration")
    print("="*60)
    print("This example demonstrates the VectorBT-enhanced components")
    print("for backtesting, financial metrics, and portfolio optimization.")
    
    try:
        # Demonstrate VectorBT backtesting
        backtesting_results = demonstrate_vectorbt_backtesting()
        
        # Demonstrate VectorBT financial metrics
        metrics_results = demonstrate_vectorbt_metrics()
        
        # Demonstrate VectorBT portfolio optimization
        optimization_results = demonstrate_vectorbt_portfolio_optimization()
        
        # Demonstrate unified vectorization manager
        demonstrate_unified_vectorization_manager()
        
        # Run performance benchmark
        run_performance_benchmark()
        
        print("\n" + "="*60)
        print("✅ VECTORBT INTEGRATION DEMONSTRATION COMPLETED")
        print("="*60)
        print("All VectorBT components are working correctly!")
        print("You can now use these components in your trading system.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demonstration failed")


if __name__ == "__main__":
    main()