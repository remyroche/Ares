"""
Test script for the refactored NAS-TAS clustering component.

This script tests the basic functionality of the refactored clustering modules.
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta

# Import the refactored component
from src.training.steps.market_analysis.clusters import (
    NASTASClusteringComponent,
    NASTASClusteringConfig
)


def create_test_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Create test market data."""
    try:
        # Create time index
        start_date = datetime.now() - timedelta(days=n_samples)
        dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')
        
        # Create synthetic market data
        np.random.seed(42)
        
        # Generate price data with regime-like patterns
        price = 100.0
        prices = []
        volumes = []
        
        for i in range(n_samples):
            # Create regime-like patterns
            if i % 200 < 100:  # Bull market regime
                price += np.random.normal(0.1, 0.5)
                volume = np.random.exponential(1000)
            else:  # Bear market regime
                price += np.random.normal(-0.05, 0.3)
                volume = np.random.exponential(800)
            
            prices.append(price)
            volumes.append(volume)
        
        # Create additional features
        returns = np.diff(prices, prepend=prices[0])
        volatility = pd.Series(returns).rolling(window=20).std().fillna(0)
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes,
            'returns': returns,
            'volatility': volatility
        })
        
        # Set timestamp as index
        market_data.set_index('timestamp', inplace=True)
        
        return market_data
        
    except Exception as e:
        print(f"Test data creation failed: {e}")
        return pd.DataFrame()


async def test_refactored_clustering():
    """Test the refactored clustering component."""
    try:
        print("🧪 Testing Refactored NAS-TAS Clustering Component")
        print("=" * 60)
        
        # Create test data
        print("📊 Creating test market data...")
        market_data = create_test_data(n_samples=500, n_features=20)
        print(f"✅ Created test data: {market_data.shape}")
        
        # Initialize configuration
        print("\n⚙️ Initializing configuration...")
        config = NASTASClusteringConfig(
            n_regimes=4,
            feature_categories=['regime_volatility', 'regime_volume'],
            use_standardized_features=True
        )
        print("✅ Configuration initialized")
        
        # Initialize component
        print("\n🔧 Initializing clustering component...")
        clustering_component = NASTASClusteringComponent(config=config)
        print("✅ Component initialized")
        
        # Test individual steps
        print("\n🔍 Testing individual steps...")
        
        # Test Step 1: Feature Preparation
        print("  📊 Testing Step 1: Feature Preparation...")
        try:
            step1_result = await clustering_component.execute_step_individually(
                "step1_feature_preparation", 
                np.random.randn(100, 10), 
                market_data
            )
            print(f"  ✅ Step 1 completed: {step1_result['success']}")
        except Exception as e:
            print(f"  ❌ Step 1 failed: {e}")
        
        # Test Step 2: Initial Clustering
        print("  🔍 Testing Step 2: Initial Clustering...")
        try:
            step2_result = await clustering_component.execute_step_individually(
                "step2_initial_clustering", 
                np.random.randn(100, 10), 
                market_data
            )
            print(f"  ✅ Step 2 completed: {step2_result['success']}")
        except Exception as e:
            print(f"  ❌ Step 2 failed: {e}")
        
        # Test full pipeline
        print("\n🚀 Testing full clustering pipeline...")
        try:
            result = await clustering_component.run(market_data)
            print("✅ Full pipeline completed successfully")
            
            # Display results summary
            if 'clustering_result' in result:
                clustering_result = result['clustering_result']
                print(f"\n📋 Results Summary:")
                print(f"  - Clusters: {clustering_result.get('clustering_result', {}).get('n_clusters', 'N/A')}")
                print(f"  - Samples: {clustering_result.get('clustering_result', {}).get('total_samples', 'N/A')}")
                print(f"  - Features: {clustering_result.get('clustering_result', {}).get('feature_count', 'N/A')}")
            
            # Display performance metrics
            if 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                print(f"\n⏱️ Performance Metrics:")
                print(f"  - Success count: {perf_metrics.get('success_count', 0)}")
                print(f"  - Error count: {perf_metrics.get('error_count', 0)}")
            
        except Exception as e:
            print(f"❌ Full pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test performance summary
        print("\n📊 Testing performance summary...")
        try:
            perf_summary = clustering_component.get_performance_summary()
            print("✅ Performance summary generated")
            print(f"  - Component: {perf_summary.get('component', 'N/A')}")
            print(f"  - Version: {perf_summary.get('version', 'N/A')}")
            print(f"  - Refactored: {perf_summary.get('refactored_architecture', False)}")
        except Exception as e:
            print(f"❌ Performance summary failed: {e}")
        
        # Test step info
        print("\n📋 Testing step information...")
        try:
            step_info = clustering_component.get_step_info()
            print("✅ Step information retrieved")
            for step_name, description in step_info.items():
                print(f"  - {step_name}: {description}")
        except Exception as e:
            print(f"❌ Step information failed: {e}")
        
        print("\n🎉 Refactored clustering component test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_modular_architecture():
    """Test the modular architecture components."""
    try:
        print("\n🔧 Testing Modular Architecture Components")
        print("=" * 60)
        
        # Test individual modules
        from src.training.steps.market_analysis.clusters import (
            FeaturePreparationStep,
            InitialClusteringStep,
            IterativeOptimization,
            ValidationStep,
            ResultsConsolidationStep,
            ClusteringOrchestrator
        )
        
        print("✅ All clustering modules imported successfully")
        
        # Test module initialization
        print("\n📊 Testing module initialization...")
        
        modules = [
            ("FeaturePreparationStep", FeaturePreparationStep),
            ("InitialClusteringStep", InitialClusteringStep),
            ("IterativeOptimization", IterativeOptimization),
            ("ValidationStep", ValidationStep),
            ("ResultsConsolidationStep", ResultsConsolidationStep),
            ("ClusteringOrchestrator", ClusteringOrchestrator)
        ]
        
        for module_name, module_class in modules:
            try:
                instance = module_class(verbose=False)
                print(f"  ✅ {module_name} initialized")
            except Exception as e:
                print(f"  ❌ {module_name} failed: {e}")
        
        print("\n🎉 Modular architecture test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Modular architecture test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting Refactored NAS-TAS Clustering Tests")
    print("=" * 80)
    
    # Run tests
    asyncio.run(test_refactored_clustering())
    asyncio.run(test_modular_architecture())
    
    print("\n🏁 All tests completed!")
    print("=" * 80)