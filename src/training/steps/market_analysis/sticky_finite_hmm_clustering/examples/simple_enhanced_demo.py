"""
Simple Enhanced Demo for Sticky Finite HMM Clustering

This demonstrates the enhanced features without complex dependencies.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer,
    StickyFiniteHMMConfig
)

def create_sample_data(n_samples: int = 1000, n_features: int = 5) -> pd.DataFrame:
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

def demo_enhanced_svi_features():
    """Demonstrate enhanced SVI features."""
    print("=" * 80)
    print("DEMO: Enhanced SVI Features")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    print(f"📊 Created sample data: {len(market_data)} rows")
    
    # Configuration with enhanced SVI features enabled
    enhanced_config = StickyFiniteHMMConfig(
        K=5,
        base_alpha=0.5,
        kappa=10.0,
        num_iters=50,  # Reduced for demo
        lr=1e-2,
        enable_pca=True,
        pca_components=5,  # Reduced for demo
        
        # Enhanced SVI Features
        enable_natural_gradients=True,
        enable_rao_blackwellization=True,
        enable_vectorization=True,
        natural_gradient_lr=0.5,
        rao_blackwell_samples=50,
        natural_gradient_frequency=5
    )
    
    print("🧠 Enhanced SVI Features:")
    print(f"   ✅ Natural Gradients: {enhanced_config.enable_natural_gradients}")
    print(f"   ✅ Rao-Blackwellization: {enhanced_config.enable_rao_blackwellization}")
    print(f"   ✅ Vectorization: {enhanced_config.enable_vectorization}")
    print(f"   ✅ Natural Gradient LR: {enhanced_config.natural_gradient_lr}")
    print(f"   ✅ Natural Gradient Frequency: {enhanced_config.natural_gradient_frequency}")
    
    # Create clusterer with enhanced features
    clusterer = StickyFiniteHMMClusterer(enhanced_config)
    
    print("\n🚀 Running Enhanced Sticky Finite HMM Clustering...")
    
    try:
        # Run clustering
        result = clusterer.fit_predict(market_data)
        
        print("\n✅ Enhanced Clustering Results:")
        print(f"   📊 Discovered {result.n_clusters} regimes")
        print(f"   🎯 Composite Score: {result.composite_score:.4f}")
        print(f"   📈 Final ELBO: {result.final_elbo:.2f}")
        
        if result.quality_assessment:
            metrics = result.quality_assessment
            print(f"   📊 Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
            print(f"   📊 Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.4f}")
            print(f"   📊 Transition Persistence: {metrics.get('transition_persistence', 0):.4f}")
        
        print(f"   ⏱️  Execution Time: {result.metadata.get('execution_time', 0):.2f}s")
        
        return result
        
    except Exception as e:
        print(f"❌ Enhanced clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_standard_vs_enhanced():
    """Compare standard vs enhanced features."""
    print("\n" + "=" * 80)
    print("DEMO: Standard vs Enhanced Features Comparison")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Standard configuration
    standard_config = StickyFiniteHMMConfig(
        K=5,
        base_alpha=0.5,
        kappa=10.0,
        num_iters=50,
        lr=1e-2,
        enable_pca=True,
        pca_components=5,
        
        # Enhanced features disabled
        enable_natural_gradients=False,
        enable_rao_blackwellization=False,
        enable_vectorization=False
    )
    
    # Enhanced configuration
    enhanced_config = StickyFiniteHMMConfig(
        K=5,
        base_alpha=0.5,
        kappa=10.0,
        num_iters=50,
        lr=1e-2,
        enable_pca=True,
        pca_components=5,
        
        # Enhanced features enabled
        enable_natural_gradients=True,
        enable_rao_blackwellization=True,
        enable_vectorization=True,
        natural_gradient_lr=0.5,
        natural_gradient_frequency=5
    )
    
    print("🔄 Running Standard Configuration...")
    standard_clusterer = StickyFiniteHMMClusterer(standard_config)
    
    try:
        standard_result = standard_clusterer.fit_predict(market_data)
        print(f"   ✅ Standard Score: {standard_result.composite_score:.4f}")
        print(f"   ⏱️  Time: {standard_result.metadata.get('execution_time', 0):.2f}s")
    except Exception as e:
        print(f"   ❌ Standard failed: {e}")
        standard_result = None
    
    print("\n🔄 Running Enhanced Configuration...")
    enhanced_clusterer = StickyFiniteHMMClusterer(enhanced_config)
    
    try:
        enhanced_result = enhanced_clusterer.fit_predict(market_data)
        print(f"   ✅ Enhanced Score: {enhanced_result.composite_score:.4f}")
        print(f"   ⏱️  Time: {enhanced_result.metadata.get('execution_time', 0):.2f}s")
    except Exception as e:
        print(f"   ❌ Enhanced failed: {e}")
        enhanced_result = None
    
    # Comparison
    if standard_result and enhanced_result:
        print("\n📊 Performance Comparison:")
        improvement = (enhanced_result.composite_score - standard_result.composite_score) / standard_result.composite_score * 100
        time_diff = enhanced_result.metadata.get('execution_time', 0) - standard_result.metadata.get('execution_time', 0)
        
        print(f"   Score Improvement: {improvement:+.2f}%")
        print(f"   Time Difference: {time_diff:+.2f}s")
        
        if improvement > 0:
            print("   🎉 Enhanced features improved clustering quality!")
        else:
            print("   ⚠️  Enhanced features did not improve quality in this case")
    
    return standard_result, enhanced_result

def demo_parameter_sweep():
    """Demonstrate parameter sweep with enhanced features."""
    print("\n" + "=" * 80)
    print("DEMO: Parameter Sweep with Enhanced Features")
    print("=" * 80)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Test different parameter combinations
    test_configs = [
        {
            'name': 'Conservative',
            'config': StickyFiniteHMMConfig(
                K=3, base_alpha=1.0, kappa=20.0, num_iters=50,
                enable_natural_gradients=True, enable_rao_blackwellization=True
            )
        },
        {
            'name': 'Balanced',
            'config': StickyFiniteHMMConfig(
                K=5, base_alpha=0.5, kappa=10.0, num_iters=50,
                enable_natural_gradients=True, enable_rao_blackwellization=True
            )
        },
        {
            'name': 'Aggressive',
            'config': StickyFiniteHMMConfig(
                K=7, base_alpha=0.2, kappa=5.0, num_iters=50,
                enable_natural_gradients=True, enable_rao_blackwellization=True
            )
        }
    ]
    
    results = {}
    
    for test_case in test_configs:
        print(f"\n🔄 Running {test_case['name']} Configuration...")
        
        try:
            clusterer = StickyFiniteHMMClusterer(test_case['config'])
            result = clusterer.fit_predict(market_data)
            
            results[test_case['name']] = result
            print(f"   ✅ Score: {result.composite_score:.4f}")
            print(f"   📊 Regimes: {result.n_clusters}")
            print(f"   ⏱️  Time: {result.metadata.get('execution_time', 0):.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[test_case['name']] = None
    
    # Summary
    print("\n📊 Parameter Sweep Summary:")
    for name, result in results.items():
        if result:
            print(f"   {name}: Score={result.composite_score:.4f}, Regimes={result.n_clusters}")
        else:
            print(f"   {name}: Failed")
    
    return results

def main():
    """Run all demos."""
    print("🚀 Simple Enhanced Sticky Finite HMM Demo")
    print("This demonstrates the enhanced SVI features:")
    print("  - Natural gradient updates for reduced variance")
    print("  - Rao-Blackwellization for exact sufficient statistics")
    print("  - Vectorized computations for optimal performance")
    
    try:
        # Run demos
        demo_enhanced_svi_features()
        demo_standard_vs_enhanced()
        demo_parameter_sweep()
        
        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
