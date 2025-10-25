"""
Test script for Automated HDBSCAN Parameter Tuner

This script demonstrates how to use the automated HDBSCAN parameter tuner
with the specific optimization targets for regime discovery.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import time

# Import the automated tuner
from src.training.steps.market_analysis.hdbscan_clustering.optimization.automated_hdbscan_parameter_tuner import (
    create_automated_hdbscan_tuner, ClusteringQualityMetrics
)

def create_sample_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Create sample financial time series data for testing."""
    np.random.seed(42)
    
    # Create synthetic financial data with multiple regimes
    data = {}
    
    # Price data with different volatility regimes
    for i in range(n_samples):
        if i < n_samples // 3:
            # High volatility regime
            data.setdefault('close', []).append(100 + np.random.normal(0, 2))
        elif i < 2 * n_samples // 3:
            # Low volatility regime
            data.setdefault('close', []).append(105 + np.random.normal(0, 0.5))
        else:
            # Medium volatility regime
            data.setdefault('close', []).append(110 + np.random.normal(0, 1))
    
    # Add technical indicators
    close_prices = np.array(data['close'])
    data['returns'] = np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]])
    data['sma_20'] = pd.Series(close_prices).rolling(20).mean().fillna(close_prices[0]).values
    data['volatility'] = pd.Series(data['returns']).rolling(20).std().fillna(0).values
    
    # Add more features
    for i in range(n_features - 4):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)

def test_automated_tuner():
    """Test the automated HDBSCAN parameter tuner with optimizations."""
    print("🧪 Testing Optimized Automated HDBSCAN Parameter Tuner")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample financial data...")
    data = create_sample_data(n_samples=500, n_features=8)
    print(f"✅ Created dataset: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Create tuner with optimizations
    print("\n🔧 Initializing optimized automated tuner...")
    tuner = create_automated_hdbscan_tuner()
    
    # Check optimization components
    print("\n🔍 Checking optimization components:")
    print(f"  • VectorBT Optimization: {'✅' if hasattr(tuner, 'vectorbt_optimizer') and tuner.vectorbt_optimizer else '❌'}")
    print(f"  • Hardware Optimization: {'✅' if hasattr(tuner, 'hardware_manager') and tuner.hardware_manager else '❌'}")
    print(f"  • Math Validation: {'✅' if hasattr(tuner, 'memory_optimizer') and tuner.memory_optimizer else '❌'}")
    
    # Run parameter tuning
    print("\n🎯 Starting optimized parameter optimization...")
    start_time = time.time()
    
    try:
        best_params, quality_metrics = tuner.tune_parameters(
            data=data,
            n_trials=20,  # Reduced for testing
            timeout=300,  # 5 minutes
            enable_fallback=True
        )
        
        optimization_time = time.time() - start_time
        
        print(f"\n✅ Optimization completed in {optimization_time:.2f} seconds")
        print(f"🏆 Best parameters: {best_params}")
        
        # Generate comprehensive report
        print("\n📊 Generating optimization report...")
        report = tuner.generate_optimization_report(best_params, quality_metrics, optimization_time)
        
        # Display results
        print("\n" + "=" * 60)
        print("📈 OPTIMIZED OPTIMIZATION RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 Target Assessment:")
        for key, value in report['target_assessment'].items():
            status = "✅" if value else "❌"
            print(f"  {status} {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 Quality Metrics:")
        for key, value in report['quality_metrics'].items():
            if value is not None:
                print(f"  • {key.replace('_', ' ').title()}: {value:.4f}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n🏆 Composite Score: {report['optimization_summary']['composite_score']:.4f}")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"  • Optimization Time: {optimization_time:.2f}s")
        print(f"  • Memory Efficiency: {'✅' if optimization_time < 60 else '⚠️'}")
        print(f"  • VectorBT Usage: {'✅' if hasattr(tuner, 'vectorbt_optimizer') and tuner.vectorbt_optimizer else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test the quality metrics calculation."""
    print("\n🧪 Testing Quality Metrics")
    print("-" * 30)
    
    # Test different quality scenarios
    scenarios = [
        {
            'name': 'Optimal Clustering',
            'metrics': ClusteringQualityMetrics(
                silhouette_score=0.6,
                calinski_harabasz_score=150.0,
                davies_bouldin_score=1.2,
                n_clusters=6,
                noise_ratio=0.1,
                within_cluster_cv=0.15,
                between_cluster_cv=0.25,
                economic_separation=0.08
            )
        },
        {
            'name': 'Poor Clustering',
            'metrics': ClusteringQualityMetrics(
                silhouette_score=-0.2,
                calinski_harabasz_score=5.0,
                davies_bouldin_score=8.0,
                n_clusters=2,
                noise_ratio=0.6,
                within_cluster_cv=0.8,
                between_cluster_cv=0.05,
                economic_separation=0.01
            )
        },
        {
            'name': 'Too Many Clusters',
            'metrics': ClusteringQualityMetrics(
                silhouette_score=0.3,
                calinski_harabasz_score=50.0,
                davies_bouldin_score=3.0,
                n_clusters=12,
                noise_ratio=0.2,
                within_cluster_cv=0.2,
                between_cluster_cv=0.15,
                economic_separation=0.06
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        metrics = scenario['metrics']
        print(f"  • Composite Score: {metrics.calculate_composite_score():.4f}")
        print(f"  • Poor Quality: {metrics.is_poor_quality()}")
        print(f"  • Clusters: {metrics.n_clusters}")
        print(f"  • Silhouette: {metrics.silhouette_score}")
        print(f"  • Within-cluster CV: {metrics.within_cluster_cv}")
        print(f"  • Between-cluster CV: {metrics.between_cluster_cv}")

if __name__ == "__main__":
    print("🚀 Automated HDBSCAN Parameter Tuner Test Suite")
    print("=" * 60)
    
    # Test quality metrics
    test_quality_metrics()
    
    # Test full optimization
    success = test_automated_tuner()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
