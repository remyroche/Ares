#!/usr/bin/env python3
"""
Balanced HMM Clustering Example

This example demonstrates how to use the enhanced HMM clustering with
cluster balancing to ensure no single cluster contains more than 15% of samples.

Key Features Demonstrated:
- Cluster size constraint enforcement
- Adaptive cluster splitting/merging
- Performance comparison with/without balancing
- Detailed cluster analysis and validation
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from market_analysis.hmm_clustering.enhanced_hmm_clustering import (
        EnhancedHMMClustering, HMMClusteringConfig
    )
    from market_analysis.hmm_clustering.cluster_balancing import (
        ClusterBalancer, ClusterBalancingConfig, BalancingMethod
    )
except ImportError as e:
    logger.error(f"Failed to import clustering modules: {e}")
    sys.exit(1)

def create_imbalanced_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Create synthetic market data with intentionally imbalanced regimes.
    This simulates the problem where one regime dominates the dataset.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create highly imbalanced regimes
    # Regime 0: 50% of data (bull market)
    # Regime 1: 30% of data (bear market) 
    # Regime 2: 15% of data (sideways)
    # Regime 3: 5% of data (high volatility)
    
    regime_proportions = [0.50, 0.30, 0.15, 0.05]
    regime_lengths = [int(n_samples * prop) for prop in regime_proportions]
    
    # Adjust to ensure exact total
    regime_lengths[-1] = n_samples - sum(regime_lengths[:-1])
    
    regimes_data = []
    prices = [100.0]
    
    for regime_id, length in enumerate(regime_lengths):
        if regime_id == 0:  # Dominant bull market (50%)
            trend = 0.0008
            volatility = 0.015
        elif regime_id == 1:  # Bear market (30%)
            trend = -0.0004
            volatility = 0.025
        elif regime_id == 2:  # Sideways market (15%)
            trend = 0.0001
            volatility = 0.012
        else:  # High volatility market (5%)
            trend = 0.0003
            volatility = 0.045
        
        # Generate returns for this regime
        regime_returns = np.random.normal(trend, volatility, length)
        regimes_data.extend(regime_returns)
        
        # Generate prices
        for ret in regime_returns:
            prices.append(prices[-1] * (1 + ret))
    
    # Ensure we have the right number of samples
    regimes_data = regimes_data[:n_samples]
    prices = prices[:n_samples]
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure OHLCV consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    logger.info(f"Created imbalanced synthetic data with expected regime distribution:")
    logger.info(f"  Regime 0 (Bull): ~50%")
    logger.info(f"  Regime 1 (Bear): ~30%")
    logger.info(f"  Regime 2 (Sideways): ~15%")
    logger.info(f"  Regime 3 (High Vol): ~5%")
    
    return data

def analyze_cluster_balance(result, title: str = "Cluster Analysis"):
    """Analyze and display cluster balance information."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Analyze regime distribution
    unique_regimes, counts = np.unique(result.regime_labels, return_counts=True)
    total_samples = len(result.regime_labels)
    
    print(f"Total samples: {total_samples:,}")
    print(f"Number of clusters: {len(unique_regimes)}")
    print(f"\nCluster Distribution:")
    
    for regime, count in zip(unique_regimes, counts):
        percentage = (count / total_samples) * 100
        status = "✅" if percentage <= 15.0 else "❌"
        print(f"  Cluster {regime}: {count:,} samples ({percentage:.2f}%) {status}")
    
    # Check balance constraints
    max_cluster_pct = max(counts) / total_samples * 100
    min_cluster_pct = min(counts) / total_samples * 100
    
    print(f"\nBalance Analysis:")
    print(f"  Largest cluster: {max_cluster_pct:.2f}%")
    print(f"  Smallest cluster: {min_cluster_pct:.2f}%")
    print(f"  Balance ratio: {min_cluster_pct/max_cluster_pct:.3f}")
    
    balance_status = "✅ BALANCED" if max_cluster_pct <= 15.0 else "❌ IMBALANCED"
    print(f"  Status: {balance_status}")
    
    # Show balancing info if available
    if hasattr(result, 'balancing_info') and result.balancing_info:
        balancing = result.balancing_info
        print(f"\nBalancing Information:")
        print(f"  Method: {balancing.get('method', 'N/A')}")
        print(f"  Balanced: {balancing.get('balanced', False)}")
        print(f"  Improvement: {balancing.get('improvement', 0):.2f}% reduction")
        print(f"  Iterations: {balancing.get('iterations', 0)}")
    
    # Performance metrics
    if hasattr(result, 'performance_metrics'):
        metrics = result.performance_metrics
        print(f"\nPerformance Metrics:")
        print(f"  Regime Balance: {metrics.get('regime_balance', 0):.4f}")
        print(f"  Regime Stability: {metrics.get('regime_stability', 0):.4f}")
        print(f"  Average Confidence: {metrics.get('avg_confidence', 0):.4f}")

def compare_balancing_methods():
    """Compare different balancing methods."""
    logger.info("🔄 Comparing different balancing methods...")
    
    # Create imbalanced synthetic data
    data = create_imbalanced_synthetic_data(5000)
    
    # Test configurations
    test_configs = [
        ("No Balancing", {"enable_cluster_balancing": False}),
        ("Hybrid Balancing", {
            "enable_cluster_balancing": True,
            "max_cluster_size_pct": 15.0,
            "cluster_balancing_method": "hybrid"
        }),
        ("Adaptive Splitting", {
            "enable_cluster_balancing": True,
            "max_cluster_size_pct": 15.0,
            "cluster_balancing_method": "adaptive_splitting"
        }),
        ("Post Processing", {
            "enable_cluster_balancing": True,
            "max_cluster_size_pct": 15.0,
            "cluster_balancing_method": "post_processing"
        })
    ]
    
    results = {}
    
    for config_name, config_params in test_configs:
        logger.info(f"\n📊 Testing: {config_name}")
        
        # Create configuration
        config = HMMClusteringConfig(
            n_components=4,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd", "bollinger_bands"],
            **config_params
        )
        
        try:
            # Initialize clustering
            clustering = EnhancedHMMClustering(config)
            
            # Engineer features
            features = clustering.engineer_features(data)
            
            # Fit model
            result = clustering.fit_hmm_model(features)
            
            results[config_name] = result
            
            # Quick analysis
            unique_regimes, counts = np.unique(result.regime_labels, return_counts=True)
            max_cluster_pct = max(counts) / len(result.regime_labels) * 100
            
            logger.info(f"  Max cluster size: {max_cluster_pct:.2f}%")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue
    
    return results

def visualize_cluster_balance(results: dict, save_path: str = None):
    """Visualize cluster balance comparison."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cluster Balance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (config_name, result) in enumerate(results.items()):
            if result is None:
                continue
                
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # Calculate cluster distribution
            unique_regimes, counts = np.unique(result.regime_labels, return_counts=True)
            percentages = (counts / len(result.regime_labels)) * 100
            
            # Create bar plot
            bars = ax.bar(unique_regimes, percentages)
            
            # Color bars based on balance (red if > 15%, green if <= 15%)
            for bar, pct in zip(bars, percentages):
                if pct > 15.0:
                    bar.set_color('red')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('green')
                    bar.set_alpha(0.7)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom')
            
            # Add 15% threshold line
            ax.axhline(y=15.0, color='orange', linestyle='--', alpha=0.8, 
                      label='15% Threshold')
            
            ax.set_title(config_name, fontweight='bold')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Percentage of Samples (%)')
            ax.set_ylim(0, max(60, max(percentages) * 1.1))
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

def run_balanced_clustering_example():
    """Run the complete balanced clustering example."""
    logger.info("🚀 Starting Balanced HMM Clustering Example")
    
    try:
        # Step 1: Compare balancing methods
        results = compare_balancing_methods()
        
        if not results:
            logger.error("No results obtained from comparison")
            return
        
        # Step 2: Detailed analysis of each method
        for config_name, result in results.items():
            if result is not None:
                analyze_cluster_balance(result, f"{config_name} Results")
        
        # Step 3: Create visualizations
        output_dir = Path("market_analysis/hmm_clustering/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz_path = output_dir / "cluster_balance_comparison.png"
        visualize_cluster_balance(results, str(viz_path))
        
        # Step 4: Summary and recommendations
        print(f"\n{'='*80}")
        print("SUMMARY AND RECOMMENDATIONS")
        print(f"{'='*80}")
        
        print("\n1. CLUSTER BALANCE ANALYSIS:")
        for config_name, result in results.items():
            if result is None:
                continue
                
            unique_regimes, counts = np.unique(result.regime_labels, return_counts=True)
            max_cluster_pct = max(counts) / len(result.regime_labels) * 100
            
            status = "✅ BALANCED" if max_cluster_pct <= 15.0 else "❌ IMBALANCED"
            print(f"   {config_name}: {max_cluster_pct:.2f}% max cluster size - {status}")
        
        print("\n2. PERFORMANCE COMPARISON:")
        for config_name, result in results.items():
            if result is None or not hasattr(result, 'performance_metrics'):
                continue
                
            metrics = result.performance_metrics
            print(f"   {config_name}:")
            print(f"     - Processing time: {result.processing_time:.2f}s")
            print(f"     - Regime balance: {metrics.get('regime_balance', 0):.4f}")
            print(f"     - Regime stability: {metrics.get('regime_stability', 0):.4f}")
        
        print("\n3. RECOMMENDATIONS:")
        print("   ✅ Use 'Hybrid Balancing' for best overall performance")
        print("   ✅ Set max_cluster_size_pct to 15.0% to prevent dominance")
        print("   ✅ Enable cluster balancing for production systems")
        print("   ✅ Monitor cluster distribution in real-time applications")
        
        logger.info("✅ Balanced clustering example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

def test_cluster_balancer_directly():
    """Test the cluster balancer directly with synthetic imbalanced data."""
    logger.info("🧪 Testing ClusterBalancer directly...")
    
    # Create synthetic imbalanced cluster data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features
    features = np.random.randn(n_samples, n_features)
    
    # Create highly imbalanced labels (cluster 0 has 60% of data)
    labels = np.array([0] * 600 + [1] * 200 + [2] * 150 + [3] * 50)
    np.random.shuffle(labels)
    
    # Create dummy probabilities
    probabilities = np.random.rand(n_samples, 4)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    logger.info("Original cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = (count / n_samples) * 100
        logger.info(f"  Cluster {cluster}: {count} samples ({pct:.2f}%)")
    
    # Test balancer
    config = ClusterBalancingConfig(
        max_cluster_size_pct=15.0,
        balancing_method=BalancingMethod.HYBRID
    )
    
    balancer = ClusterBalancer(config)
    balanced_labels, balanced_probs, balancing_info = balancer.balance_clusters(
        features, labels, probabilities
    )
    
    logger.info("\nBalanced cluster distribution:")
    unique, counts = np.unique(balanced_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = (count / n_samples) * 100
        logger.info(f"  Cluster {cluster}: {count} samples ({pct:.2f}%)")
    
    logger.info(f"\nBalancing info: {balancing_info}")
    
    # Validate balance
    validation = balancer.validate_balance(balanced_labels)
    logger.info(f"Balance validation: {validation}")

if __name__ == "__main__":
    # Run the complete example
    run_balanced_clustering_example()
    
    # Test the balancer directly
    print("\n" + "="*80)
    test_cluster_balancer_directly()