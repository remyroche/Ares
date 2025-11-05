"""
Enhanced Auto-Tuning Example for Sticky Finite HMM Clustering

This example demonstrates the enhanced standalone runner with:
- 2-stage optimization (grid search -> fine grid search)
- Multi-objective optimization
- Quality assessor integration
- Composite scoring and KPI tracking
- Enhanced SVI optimizations (natural gradients, Rao-Blackwellization)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.enhanced_standalone_runner import (
    run_sticky_finite_hmm_with_auto_tuning,
    AutoTuningConfig,
    OptimizationResult
)

def create_sample_data(n_samples: int = 2000, n_features: int = 10) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    timestamps = pd.date_range(start="2022-01-01", periods=n_samples, freq="1h")
    
    # Simulate price movements with regime changes
    price = 100.0
    prices = [price]
    
    for i in range(1, n_samples):
        # Random walk with occasional regime changes
        if np.random.random() < 0.02:  # 2% chance of regime change
            volatility = np.random.uniform(0.001, 0.01)
        else:
            volatility = 0.002
        
        change = np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.005, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.005, n_samples)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    return data

def example_basic_auto_tuning():
    """Example: Basic 2-stage auto-tuning."""
    print("=" * 80)
    print("ENHANCED AUTO-TUNING EXAMPLE: Basic 2-Stage Optimization")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    print(f"📊 Created sample data: {len(market_data)} rows")
    
    # Run basic auto-tuning
    result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=market_data,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        optimization_stages=2,  # grid -> fine grid
        use_multi_objective=False,
        max_trials_per_stage=20,  # Reduced for demo
        enable_kpi_tracking=True,
        save_results=False  # Don't save artifacts in demo
    )
    
    print("\n🎯 Basic Auto-Tuning Results:")
    print(f"   Best Score: {result.best_score:.4f}")
    print(f"   Best Parameters: {result.best_params}")
    print(f"   Optimization Time: {result.optimization_time:.2f}s")
    print(f"   Total Trials: {len(result.all_trials)}")
    
    if result.kpi_metrics:
        print(f"   Success Rate: {result.kpi_metrics.get('success_rate', 0):.2%}")
        print(f"   Trials/Second: {result.kpi_metrics.get('trials_per_second', 0):.2f}")
    
    return result

def example_multi_objective_optimization():
    """Example: Multi-objective optimization."""
    print("\n" + "=" * 80)
    print("ENHANCED AUTO-TUNING EXAMPLE: Multi-Objective Optimization")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Run multi-objective optimization
    result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=market_data,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        optimization_stages=2,
        use_multi_objective=True,
        objectives=["composite_score", "silhouette_score", "transition_persistence"],
        max_trials_per_stage=15,  # Reduced for demo
        enable_kpi_tracking=True,
        save_results=False
    )
    
    print("\n🎯 Multi-Objective Results:")
    print(f"   Best Composite Score: {result.best_score:.4f}")
    print(f"   Best Objectives: {result.best_objectives}")
    print(f"   Pareto Solutions: {len(result.pareto_solutions) if result.pareto_solutions else 0}")
    
    if result.pareto_solutions:
        print("   Top 3 Pareto Solutions:")
        for i, solution in enumerate(result.pareto_solutions[:3]):
            print(f"     {i+1}. Score: {solution.score:.4f}, Objectives: {solution.objectives}")
    
    return result

def example_enhanced_svi_features():
    """Example: Enhanced SVI features demonstration."""
    print("\n" + "=" * 80)
    print("ENHANCED AUTO-TUNING EXAMPLE: Enhanced SVI Features")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Configuration with enhanced SVI features
    config = AutoTuningConfig(
        optimization_stages=2,
        use_multi_objective=False,
        max_trials_per_stage=10,
        enable_kpi_tracking=True,
        save_all_trials=True
    )
    
    # Run with enhanced features
    result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=market_data,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        auto_tuning_config=config,
        save_results=False
    )
    
    print("\n🧠 Enhanced SVI Features Results:")
    print(f"   Best Score: {result.best_score:.4f}")
    print(f"   Stage Results: {len(result.stage_results)} stages")
    
    for stage in result.stage_results:
        print(f"   Stage {stage['stage']} ({stage['stage_name']}):")
        print(f"     Trials: {stage['trials_evaluated']}")
        print(f"     Success Rate: {stage['successful_trials']/stage['trials_evaluated']:.2%}")
        print(f"     Best Score: {stage['best_score']:.4f}")
        print(f"     Stage Time: {stage['stage_time']:.2f}s")
    
    return result

def example_performance_comparison():
    """Example: Performance comparison with/without enhancements."""
    print("\n" + "=" * 80)
    print("ENHANCED AUTO-TUNING EXAMPLE: Performance Comparison")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Standard configuration (minimal enhancements)
    standard_config = AutoTuningConfig(
        optimization_stages=1,  # Only grid search
        use_multi_objective=False,
        max_trials_per_stage=20,
        enable_kpi_tracking=True
    )
    
    # Enhanced configuration
    enhanced_config = AutoTuningConfig(
        optimization_stages=2,  # grid -> fine grid
        use_multi_objective=True,
        objectives=["composite_score", "silhouette_score"],
        max_trials_per_stage=15,
        enable_kpi_tracking=True
    )
    
    print("🔄 Running standard optimization...")
    standard_result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=market_data,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        auto_tuning_config=standard_config,
        save_results=False
    )
    
    print("🔄 Running enhanced optimization...")
    enhanced_result = run_sticky_finite_hmm_with_auto_tuning(
        market_data=market_data,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h",
        auto_tuning_config=enhanced_config,
        save_results=False
    )
    
    print("\n📊 Performance Comparison:")
    print(f"   Standard:")
    print(f"     Best Score: {standard_result.best_score:.4f}")
    print(f"     Optimization Time: {standard_result.optimization_time:.2f}s")
    print(f"     Total Trials: {len(standard_result.all_trials)}")
    
    print(f"   Enhanced:")
    print(f"     Best Score: {enhanced_result.best_score:.4f}")
    print(f"     Optimization Time: {enhanced_result.optimization_time:.2f}s")
    print(f"     Total Trials: {len(enhanced_result.all_trials)}")
    
    improvement = (enhanced_result.best_score - standard_result.best_score) / standard_result.best_score * 100
    print(f"   Score Improvement: {improvement:+.2f}%")
    
    return standard_result, enhanced_result

def main():
    """Run all examples."""
    print("🚀 Enhanced Sticky Finite HMM Auto-Tuning Examples")
    print("This demonstrates the enhanced standalone runner capabilities:")
    print("  - 2-stage optimization (grid -> fine grid)")
    print("  - Multi-objective optimization")
    print("  - Quality assessor integration")
    print("  - KPI tracking and performance metrics")
    print("  - Enhanced SVI optimizations")
    
    try:
        # Run examples
        example_basic_auto_tuning()
        example_multi_objective_optimization()
        example_enhanced_svi_features()
        example_performance_comparison()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
