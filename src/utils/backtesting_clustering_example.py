"""
Example demonstrating backtesting-enhanced clustering for SR levels.

This script shows how the backtesting engine learns quality rules from historical data
and uses them to improve SR level clustering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel
from .backtesting_enhanced_clustering import BacktestingEnhancedClustering, BacktestingEnhancedConfig

def create_sample_data(days: int = 30) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data with some SR levels
    base_price = 100.0
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days*24, freq='H')
    
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(dates):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i * 0.1)  # Slow trend
        volatility = 0.02 * np.random.normal()  # 2% volatility
        
        # Create some support/resistance levels
        if i % 100 == 0:  # Every ~4 days
            # Create a support level
            support_price = current_price * 0.98
            # Price bounces off support
            current_price = support_price + abs(np.random.normal(0, 0.01)) * current_price
        elif i % 150 == 0:  # Every ~6 days
            # Create a resistance level
            resistance_price = current_price * 1.02
            # Price bounces off resistance
            current_price = resistance_price - abs(np.random.normal(0, 0.01)) * current_price
        else:
            # Normal price movement
            current_price = current_price * (1 + trend + volatility)
        
        # Generate OHLC from current price
        high = current_price * (1 + abs(np.random.normal(0, 0.005)))
        low = current_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        close_price = current_price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def create_sample_sr_levels() -> list:
    """Create sample SR levels for testing."""
    levels = []
    
    # Create some support levels
    support_prices = [95.0, 97.5, 98.0, 99.5, 101.0]
    for price in support_prices:
        level = {
            'price': price,
            'type': 'support',
            'strength': np.random.uniform(0.3, 0.9),
            'touches': np.random.randint(2, 8),
            'detection_time': datetime.now() - timedelta(days=np.random.randint(1, 20)),
            'metadata': {'detection_method': 'fractal'}
        }
        levels.append(level)
    
    # Create some resistance levels
    resistance_prices = [102.0, 103.5, 105.0, 106.5, 108.0]
    for price in resistance_prices:
        level = {
            'price': price,
            'type': 'resistance',
            'strength': np.random.uniform(0.3, 0.9),
            'touches': np.random.randint(2, 8),
            'detection_time': datetime.now() - timedelta(days=np.random.randint(1, 20)),
            'metadata': {'detection_method': 'pivot'}
        }
        levels.append(level)
    
    return levels

def demonstrate_backtesting_engine():
    """Demonstrate the backtesting engine."""
    print("🔬 Demonstrating SR Backtesting Engine")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data(30)
    levels = create_sample_sr_levels()
    
    print(f"Created {len(data)} data points and {len(levels)} SR levels")
    
    # Initialize backtesting engine
    config = BacktestConfig(
        touch_tolerance=0.002,  # 0.2% tolerance
        min_bounce_strength=0.001,  # 0.1% minimum bounce
        max_hold_time=12,  # 12 hours max hold
        success_rate_weight=0.3,
        bounce_strength_weight=0.25,
        volume_confirmation_weight=0.2,
        time_persistence_weight=0.15,
        touch_frequency_weight=0.1
    )
    
    engine = SRBacktestingEngine(config)
    
    # Convert levels to SRLevel objects
    sr_levels = []
    for level_dict in levels:
        sr_level = SRLevel(
            price=level_dict['price'],
            level_type=level_dict['type'],
            strength=level_dict['strength'],
            touches=level_dict['touches'],
            detection_time=level_dict['detection_time'],
            metadata=level_dict['metadata']
        )
        sr_levels.append(sr_level)
    
    # Backtest levels
    print("\n📊 Backtesting SR levels...")
    results = engine.backtest_multiple_levels(sr_levels, data)
    
    # Display results
    print(f"\n📈 Backtesting Results:")
    print(f"Total levels tested: {len(results)}")
    print(f"Average quality score: {np.mean([r.quality_score for r in results]):.3f}")
    print(f"Average success rate: {np.mean([r.success_rate for r in results]):.3f}")
    print(f"Average bounce strength: {np.mean([r.avg_bounce_strength for r in results]):.3f}")
    
    # Show individual results
    print(f"\n📋 Individual Level Results:")
    for i, result in enumerate(results[:5]):  # Show first 5
        print(f"Level {i+1}: ${result.level.price:.2f} ({result.level.level_type})")
        print(f"  Quality: {result.quality_score:.3f}")
        print(f"  Success Rate: {result.success_rate:.3f}")
        print(f"  Touches: {result.total_touches}")
        print(f"  Bounce Strength: {result.avg_bounce_strength:.3f}")
        print()
    
    # Learn quality rules
    print("🧠 Learning quality rules...")
    rules = engine.learn_quality_rules(results)
    
    if rules:
        print(f"✅ Learned rules with {len(rules.get('discriminative_features', {}))} key features")
        print(f"Quality threshold: {rules.get('quality_threshold', 0.0):.3f}")
        
        # Show discriminative features
        features = rules.get('discriminative_features', {})
        if features:
            print(f"\n🔍 Key discriminative features:")
            for feature, info in features.items():
                print(f"  {feature}: discriminative power = {info['discriminative_power']:.3f}")
    
    return engine, results

def demonstrate_enhanced_clustering():
    """Demonstrate backtesting-enhanced clustering."""
    print("\n🔗 Demonstrating Backtesting-Enhanced Clustering")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data(30)
    levels = create_sample_sr_levels()
    
    # Initialize enhanced clustering
    config = BacktestingEnhancedConfig(
        proximity_threshold=0.01,  # 1% of price range
        strength_similarity_threshold=0.2,  # 20% strength difference
        min_quality_score=0.3,  # Minimum quality to keep
        quality_weight_in_clustering=0.4,  # Weight of quality in clustering
        min_levels_for_learning=5  # Lower threshold for demo
    )
    
    clustering = BacktestingEnhancedClustering(config)
    
    # Get price range
    prices = [level['price'] for level in levels]
    price_range = (min(prices), max(prices))
    
    print(f"Clustering {len(levels)} levels in price range ${price_range[0]:.2f} - ${price_range[1]:.2f}")
    
    # Perform clustering
    result = clustering.cluster_with_backtesting(levels, data, price_range)
    
    # Display results
    print(f"\n📊 Clustering Results:")
    print(f"Algorithm used: {result.algorithm_used}")
    print(f"Quality score: {result.quality_score:.3f}")
    print(f"Total clusters: {len(result.clusters)}")
    print(f"Quality enhanced: {getattr(result, 'quality_enhanced', False)}")
    
    # Show cluster details
    print(f"\n🔍 Cluster Details:")
    for i, cluster in enumerate(result.clusters):
        if len(cluster) > 1:
            cluster_levels = [levels[j] for j in cluster]
            cluster_prices = [level['price'] for level in cluster_levels]
            cluster_types = [level['type'] for level in cluster_levels]
            
            print(f"Cluster {i+1}: {len(cluster)} levels")
            print(f"  Prices: {[f'${p:.2f}' for p in cluster_prices]}")
            print(f"  Types: {cluster_types}")
            print(f"  Price spread: ${max(cluster_prices) - min(cluster_prices):.2f}")
            print()
        else:
            level = levels[cluster[0]]
            print(f"Single level: ${level['price']:.2f} ({level['type']})")
    
    # Show learning summary
    learning_summary = clustering.get_learning_summary()
    print(f"\n🧠 Learning Summary:")
    print(f"Levels processed: {learning_summary['levels_processed']}")
    print(f"Rules learned: {learning_summary['rules_learned']}")
    print(f"Learned features: {learning_summary['learned_features']}")
    
    return clustering, result

def main():
    """Main demonstration function."""
    print("🚀 SR Backtesting-Enhanced Clustering Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate backtesting engine
        engine, backtest_results = demonstrate_backtesting_engine()
        
        # Demonstrate enhanced clustering
        clustering, clustering_result = demonstrate_enhanced_clustering()
        
        print("\n✅ Demonstration completed successfully!")
        print("\nKey Benefits:")
        print("• Data-driven quality assessment through backtesting")
        print("• Adaptive clustering based on learned quality rules")
        print("• Improved SR level selection and grouping")
        print("• Continuous learning from historical performance")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()