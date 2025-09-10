"""
Example script demonstrating weight optimization for SR quality score parameters.

This script shows how to:
1. Create sample SR levels and market data
2. Run backtesting to get quality scores
3. Optimize weights using different methods
4. Validate the optimized weights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.sr_clustering import (
    get_backtesting_engine, BacktestConfig, SRLevel,
    get_backtesting_enhanced_clustering, BacktestingEnhancedConfig,
    get_weight_optimization_engine, WeightOptimizationConfig
)

def create_sample_market_data(days: int = 100) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate realistic price data with some volatility
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume data
    volumes = np.random.lognormal(15, 0.5, days)  # Log-normal volume distribution
    
    return pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })

def create_sample_sr_levels() -> list:
    """Create sample SR levels for testing."""
    levels = []
    
    # Create some realistic SR levels
    base_prices = [95.0, 100.0, 105.0, 110.0, 115.0]
    
    for i, price in enumerate(base_prices):
        level = SRLevel(
            price=price,
            level_type='support' if i % 2 == 0 else 'resistance',
            strength=0.5 + (i * 0.1),
            first_touch=datetime.now() - timedelta(days=50),
            last_touch=datetime.now() - timedelta(days=5),
            touch_count=3 + i,
            timeframe='1D',
            symbol='TEST',
            source='example'
        )
        levels.append(level)
    
    return levels

def demonstrate_weight_optimization():
    """Demonstrate weight optimization for SR quality scores."""
    print("🚀 SR Weight Optimization Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample market data and SR levels...")
    market_data = create_sample_market_data(100)
    sr_levels = create_sample_sr_levels()
    
    print(f"✅ Created {len(market_data)} days of market data")
    print(f"✅ Created {len(sr_levels)} SR levels")
    
    # Initialize backtesting engine
    print("\n🔧 Initializing backtesting engine...")
    backtest_config = BacktestConfig(
        touch_tolerance=0.002,  # 0.2% tolerance
        min_touches=2,
        success_rate_weight=0.3,
        bounce_strength_weight=0.25,
        volume_confirmation_weight=0.2,
        time_persistence_weight=0.15,
        touch_frequency_weight=0.1
    )
    
    backtesting_engine = get_backtesting_engine(backtest_config)
    
    # Run backtesting
    print("\n📈 Running backtesting on SR levels...")
    backtest_results = backtesting_engine.backtest_multiple_levels(sr_levels, market_data)
    
    print(f"✅ Backtesting completed. Results: {len(backtest_results)}")
    for i, result in enumerate(backtest_results):
        print(f"   Level {i+1}: Quality Score = {result.quality_score:.3f}")
    
    # Test different optimization methods
    optimization_methods = ['scipy_minimize', 'grid_search']
    
    for method in optimization_methods:
        print(f"\n🎯 Testing {method} optimization...")
        
        # Configure weight optimization
        weight_config = WeightOptimizationConfig(
            optimization_method=method,
            primary_objective='r2_score',
            secondary_objective='stability',
            max_iterations=50 if method == 'scipy_minimize' else 1000
        )
        
        weight_optimizer = get_weight_optimization_engine(weight_config)
        
        # Run optimization
        optimization_result = weight_optimizer.optimize_weights(backtest_results, market_data)
        
        if optimization_result and optimization_result.get('optimization_success', False):
            print(f"✅ {method} optimization successful!")
            print(f"   Best Score: {optimization_result.get('best_score', 0.0):.4f}")
            print(f"   Best Weights:")
            
            best_weights = optimization_result.get('best_weights', {})
            for feature, weight in best_weights.items():
                print(f"     {feature}: {weight:.3f}")
            
            # Validate weights
            print(f"\n🔍 Validating optimized weights...")
            validation_result = weight_optimizer.validate_weights(best_weights, backtest_results)
            
            if validation_result:
                print(f"   Validation R² Score: {validation_result.get('r2_score', 0.0):.4f}")
                print(f"   Validation MSE: {validation_result.get('mse', 0.0):.4f}")
                print(f"   Validation Correlation: {validation_result.get('correlation', 0.0):.4f}")
        else:
            print(f"❌ {method} optimization failed")
    
    # Demonstrate integration with backtesting engine
    print(f"\n🔗 Testing integration with backtesting engine...")
    
    # Learn rules with weight optimization
    learned_rules = backtesting_engine.learn_quality_rules(
        backtest_results, 
        optimize_weights=True, 
        market_data=market_data
    )
    
    if learned_rules and learned_rules.get('weight_optimization_enabled', False):
        optimized_weights = learned_rules.get('optimized_weights', {})
        if optimized_weights:
            print("✅ Weight optimization integrated successfully!")
            print("   Optimized weights from backtesting engine:")
            for feature, weight in optimized_weights.items():
                print(f"     {feature}: {weight:.3f}")
        else:
            print("⚠️  Weight optimization attempted but no weights available")
    else:
        print("❌ Weight optimization integration failed")
    
    # Test clustering with optimized weights
    print(f"\n🎯 Testing clustering with optimized weights...")
    
    clustering_config = BacktestingEnhancedConfig(
        min_levels_for_learning=3,
        quality_filter_threshold=0.1,
        proximity_adjustment_factor=0.5
    )
    
    clustering = get_backtesting_enhanced_clustering(clustering_config)
    
    # Convert SR levels to dict format for clustering
    levels_dict = []
    for level in sr_levels:
        level_dict = {
            'price': level.price,
            'strength': level.strength,
            'level_type': level.level_type,
            'touch_count': level.touch_count,
            'first_touch': level.first_touch,
            'last_touch': level.last_touch
        }
        levels_dict.append(level_dict)
    
    # Run clustering
    clustering_result = clustering.cluster_with_backtesting(levels_dict, market_data)
    
    if clustering_result:
        print(f"✅ Clustering completed successfully!")
        print(f"   Number of clusters: {len(clustering_result.clusters)}")
        print(f"   Algorithm used: {clustering_result.algorithm_used}")
        
        for i, cluster in enumerate(clustering_result.clusters):
            print(f"   Cluster {i+1}: {len(cluster)} levels, avg quality: {cluster.get('avg_quality', 0.0):.3f}")
    else:
        print("❌ Clustering failed")
    
    print(f"\n🎉 Weight optimization demo completed!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        demonstrate_weight_optimization()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()