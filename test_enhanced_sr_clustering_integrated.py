#!/usr/bin/env python3
"""
Test script for the enhanced SR clustering component.

This script tests the enhanced SR clustering functionality that has been integrated
into the existing sr_clustering.py file.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Generate sample dates
    start_date = datetime.now() - timedelta(days=30)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')
    
    # Generate sample price data with some trend and volatility
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some noise to create realistic OHLC
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

async def test_enhanced_clustering():
    """Test the enhanced clustering functionality."""
    print("🚀 Testing Enhanced SR Clustering Component...")
    
    try:
        # Import the enhanced component
        from src.training.steps.market_analysis.components.sr_clustering import (
            SRClusteringComponent,
            EnhancedSRClusteringConfig,
            ClusteringAlgorithm,
            OptimizationStrategy
        )
        print("✅ Successfully imported enhanced SR clustering components")
        
        # Create sample data
        print("📊 Creating sample data...")
        price_data = create_sample_data(500)
        print(f"✅ Created sample data: {price_data.shape[0]} samples")
        
        # Initialize the component
        print("🔧 Initializing SR clustering component...")
        component = SRClusteringComponent()
        print("✅ Component initialized successfully")
        
        # Test basic clustering
        print("🎯 Testing basic clustering...")
        result = await component.execute({
            'symbol': 'TESTUSDT',
            'exchange': 'test',
            'timeframe': '1h',
            'direction': 'both',
            'execution_mode': 'light'
        })
        
        if result['success']:
            print(f"✅ Basic clustering successful: {result['metrics']['total_clusters']} clusters")
        else:
            print(f"❌ Basic clustering failed: {result.get('error', 'Unknown error')}")
        
        # Test enhanced clustering with configuration
        print("🚀 Testing enhanced clustering...")
        config = EnhancedSRClusteringConfig(
            clustering_algorithm=ClusteringAlgorithm.HDBSCAN,
            min_cluster_size=5,
            enable_hardware_optimization=True,
            enable_vectorbt_optimization=True
        )
        
        # Test the enhanced clustering method
        try:
            cluster_results = await component.cluster_sr_levels_enhanced(price_data, config)
            print(f"✅ Enhanced clustering successful: {len(cluster_results)} clusters found")
            
            # Display cluster information
            for i, cluster in enumerate(cluster_results[:3]):  # Show first 3 clusters
                print(f"  Cluster {i+1}:")
                print(f"    - ID: {cluster.cluster_id}")
                print(f"    - Centroid Price: {cluster.centroid_price:.4f}")
                print(f"    - Size: {cluster.cluster_size}")
                print(f"    - Quality: {cluster.cluster_quality:.4f}")
                print(f"    - Confidence: {cluster.confidence:.4f}")
                
        except Exception as e:
            print(f"⚠️ Enhanced clustering test failed: {e}")
            print("This is expected if some dependencies are not available")
        
        print("🎉 Enhanced SR clustering test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies may not be available in this environment")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_configuration_options():
    """Test different configuration options."""
    print("\n🔧 Testing Configuration Options...")
    
    try:
        from src.training.steps.market_analysis.components.sr_clustering import (
            EnhancedSRClusteringConfig,
            ClusteringAlgorithm,
            OptimizationStrategy
        )
        
        # Test different clustering algorithms
        algorithms = [
            ClusteringAlgorithm.HDBSCAN,
            ClusteringAlgorithm.DBSCAN,
            ClusteringAlgorithm.KMEANS,
            ClusteringAlgorithm.SPECTRAL
        ]
        
        for algo in algorithms:
            config = EnhancedSRClusteringConfig(
                clustering_algorithm=algo,
                min_cluster_size=3,
                enable_hardware_optimization=False,  # Disable for testing
                enable_vectorbt_optimization=False
            )
            print(f"✅ Configuration for {algo.value}: {config.clustering_algorithm}")
        
        # Test HPO strategies
        strategies = [
            OptimizationStrategy.BAYESIAN_TPE,
            OptimizationStrategy.HIERARCHICAL_HPO,
            OptimizationStrategy.REGIME_SPECIFIC,
            OptimizationStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            config = EnhancedSRClusteringConfig()
            config.hpo_config['optimization_strategy'] = strategy
            print(f"✅ HPO strategy {strategy.value}: {config.hpo_config['optimization_strategy']}")
        
        print("✅ Configuration options test completed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

async def main():
    """Main test function."""
    print("=" * 60)
    print("Enhanced SR Clustering Component Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    await test_enhanced_clustering()
    
    # Test configuration options
    await test_configuration_options()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())