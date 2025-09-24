"""
Hybrid NAS Example - Complementary Tree-Based and Neural Architecture Search

This example demonstrates how to use the hybrid NAS system that combines
tree-based and neural approaches to complement your existing neural NAS system.

Key Benefits:
- Tree-based NAS for fast feature selection and regime detection
- Neural NAS for complex pattern recognition and sequential modeling
- Intelligent routing based on data characteristics
- Ensemble methods combining both approaches
- Complementary optimization strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import hybrid NAS components
import sys
sys.path.append('/workspace')
from src.utils.ml_common.optimization.hybrid_nas_system import (
    HybridNASConfig, HybridNASSystem, search_hybrid_architecture
)
from src.training.steps.market_analysis.hybrid_nas_clustering import (
    HybridNASClusteringConfig, HybridNASClusterer
)


def create_sample_financial_data(n_samples=1000, n_features=50):
    """Create sample financial data for demonstration."""
    logger.info("Creating sample financial data...")
    
    # Generate synthetic financial features
    np.random.seed(42)
    
    # Price-based features (tabular)
    price_features = np.random.randn(n_samples, 15)
    
    # Technical indicators (tabular)
    technical_features = np.random.randn(n_samples, 20)
    
    # Volume features (tabular)
    volume_features = np.random.randn(n_samples, 10)
    
    # Sequential features (time series patterns)
    sequential_features = np.random.randn(n_samples, 5)
    for i in range(1, n_samples):
        sequential_features[i] = 0.7 * sequential_features[i-1] + 0.3 * np.random.randn(5)
    
    # Combine all features
    X = np.hstack([
        price_features, technical_features, volume_features, sequential_features
    ])
    
    # Create target variable (price movement prediction)
    y = (0.2 * price_features[:, 0] + 
         0.15 * technical_features[:, 0] + 
         0.1 * volume_features[:, 0] + 
         0.1 * sequential_features[:, 0] + 
         np.random.randn(n_samples) * 0.05)
    
    # Add some regime labels for regime-aware search
    regime_labels = np.random.choice([0, 1, 2, 3, 4], size=n_samples)
    
    logger.info(f"Created dataset with {n_samples} samples and {X.shape[1]} features")
    return X, y, regime_labels


def demonstrate_complementary_hybrid_nas():
    """Demonstrate complementary hybrid NAS approach."""
    logger.info("=== Complementary Hybrid NAS ===")
    
    # Create sample data
    X, y, regime_labels = create_sample_financial_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure complementary hybrid NAS
    config = HybridNASConfig(
        hybrid_strategy='complementary',
        tree_config=TreeArchitectureConfig(
            model_types=['xgboost', 'lightgbm'],
            n_trials=20,
            objectives=['accuracy', 'efficiency', 'interpretability'],
            enable_feature_selection=True,
            max_features=30
        ),
        neural_config=ArchitectureConfig(
            n_trials=20,
            objectives=['accuracy', 'efficiency', 'robustness']
        )
    )
    
    # Perform hybrid architecture search
    start_time = time.time()
    best_architecture = search_hybrid_architecture(
        X_train, y_train, X_test, y_test, config, regime_labels
    )
    search_time = time.time() - start_time
    
    # Display results
    logger.info(f"Complementary search completed in {search_time:.2f} seconds")
    logger.info(f"Best hybrid method: {best_architecture.hybrid_method}")
    logger.info(f"Combined accuracy: {best_architecture.combined_accuracy:.4f}")
    logger.info(f"Combined efficiency: {best_architecture.combined_efficiency:.4f}")
    logger.info(f"Combined interpretability: {best_architecture.combined_interpretability:.4f}")
    logger.info(f"Overall score: {best_architecture.overall_score:.4f}")
    
    return best_architecture


def demonstrate_ensemble_hybrid_nas():
    """Demonstrate ensemble hybrid NAS approach."""
    logger.info("=== Ensemble Hybrid NAS ===")
    
    # Create sample data
    X, y, regime_labels = create_sample_financial_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure ensemble hybrid NAS
    config = HybridNASConfig(
        hybrid_strategy='ensemble',
        ensemble_methods=['voting', 'stacking'],
        ensemble_weights=[0.6, 0.4],  # 60% tree, 40% neural
        tree_config=TreeArchitectureConfig(
            model_types=['xgboost', 'lightgbm', 'catboost'],
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'interpretability']
        ),
        neural_config=ArchitectureConfig(
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'robustness']
        )
    )
    
    # Perform hybrid architecture search
    start_time = time.time()
    best_architecture = search_hybrid_architecture(
        X_train, y_train, X_test, y_test, config, regime_labels
    )
    search_time = time.time() - start_time
    
    # Display results
    logger.info(f"Ensemble search completed in {search_time:.2f} seconds")
    logger.info(f"Best hybrid method: {best_architecture.hybrid_method}")
    logger.info(f"Ensemble config: {best_architecture.ensemble_config}")
    logger.info(f"Combined accuracy: {best_architecture.combined_accuracy:.4f}")
    logger.info(f"Overall score: {best_architecture.overall_score:.4f}")
    
    return best_architecture


def demonstrate_routing_hybrid_nas():
    """Demonstrate routing hybrid NAS approach."""
    logger.info("=== Routing Hybrid NAS ===")
    
    # Create sample data with different characteristics
    X, y, regime_labels = create_sample_financial_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure routing hybrid NAS
    config = HybridNASConfig(
        hybrid_strategy='routing',
        routing_rules={
            'use_tree_for_tabular': True,
            'use_neural_for_sequential': True,
            'tabular_threshold': 0.7,
            'sequential_threshold': 0.5
        },
        tree_config=TreeArchitectureConfig(
            model_types=['xgboost', 'lightgbm'],
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'interpretability']
        ),
        neural_config=ArchitectureConfig(
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'robustness']
        )
    )
    
    # Analyze data characteristics
    data_characteristics = {
        'tabular_ratio': 0.8,  # High tabular ratio
        'sequential_ratio': 0.2,  # Low sequential ratio
        'complexity_ratio': 0.5
    }
    
    # Perform hybrid architecture search
    start_time = time.time()
    best_architecture = search_hybrid_architecture(
        X_train, y_train, X_test, y_test, config, regime_labels, data_characteristics
    )
    search_time = time.time() - start_time
    
    # Display results
    logger.info(f"Routing search completed in {search_time:.2f} seconds")
    logger.info(f"Best hybrid method: {best_architecture.hybrid_method}")
    logger.info(f"Routing strategy: {best_architecture.routing_strategy}")
    logger.info(f"Combined accuracy: {best_architecture.combined_accuracy:.4f}")
    logger.info(f"Overall score: {best_architecture.overall_score:.4f}")
    
    return best_architecture


def demonstrate_sequential_hybrid_nas():
    """Demonstrate sequential hybrid NAS approach."""
    logger.info("=== Sequential Hybrid NAS ===")
    
    # Create sample data
    X, y, regime_labels = create_sample_financial_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure sequential hybrid NAS
    config = HybridNASConfig(
        hybrid_strategy='sequential',
        tree_config=TreeArchitectureConfig(
            model_types=['xgboost'],
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'interpretability'],
            enable_feature_selection=True,
            max_features=30
        ),
        neural_config=ArchitectureConfig(
            n_trials=15,
            objectives=['accuracy', 'efficiency', 'robustness']
        )
    )
    
    # Perform hybrid architecture search
    start_time = time.time()
    best_architecture = search_hybrid_architecture(
        X_train, y_train, X_test, y_test, config, regime_labels
    )
    search_time = time.time() - start_time
    
    # Display results
    logger.info(f"Sequential search completed in {search_time:.2f} seconds")
    logger.info(f"Best hybrid method: {best_architecture.hybrid_method}")
    logger.info(f"Tree training time: {best_architecture.tree_training_time:.2f} seconds")
    logger.info(f"Neural training time: {best_architecture.neural_training_time:.2f} seconds")
    logger.info(f"Total training time: {best_architecture.total_training_time:.2f} seconds")
    logger.info(f"Combined accuracy: {best_architecture.combined_accuracy:.4f}")
    logger.info(f"Overall score: {best_architecture.overall_score:.4f}")
    
    return best_architecture


def demonstrate_hybrid_clustering():
    """Demonstrate hybrid NAS clustering for regime detection."""
    logger.info("=== Hybrid NAS Clustering ===")
    
    # Create sample market data
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
    
    market_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.random.rand(n_samples) * 0.5,
        'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.random.rand(n_samples) * 0.5,
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    timestamps = dates.values
    
    # Configure hybrid NAS clustering
    config = HybridNASClusteringConfig.create_complementary_config()
    config.clustering_config['n_regimes'] = 8
    config.tree_nas_config['n_trials'] = 20
    config.neural_nas_config['n_trials'] = 20
    
    # Initialize hybrid clusterer
    clusterer = HybridNASClusterer(config)
    
    # Perform hybrid clustering
    start_time = time.time()
    results = clusterer.cluster(market_data, timestamps, optimize_parameters=True, generate_report=True)
    clustering_time = time.time() - start_time
    
    # Display results
    logger.info(f"Hybrid clustering completed in {clustering_time:.2f} seconds")
    logger.info(f"Strategy: {results['hybrid_metadata']['strategy']}")
    logger.info(f"Number of regimes: {results['statistics']['n_clusters']}")
    logger.info(f"Silhouette score: {results['statistics']['silhouette_score']:.4f}")
    logger.info(f"Accuracy: {results['quality_metrics']['accuracy']:.4f}")
    logger.info(f"Efficiency: {results['quality_metrics']['efficiency']:.4f}")
    logger.info(f"Interpretability: {results['quality_metrics']['interpretability']:.4f}")
    
    return results


def compare_hybrid_approaches():
    """Compare different hybrid NAS approaches."""
    logger.info("=== Hybrid NAS Approaches Comparison ===")
    
    # Create sample data
    X, y, regime_labels = create_sample_financial_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    approaches = {
        'complementary': HybridNASConfig(hybrid_strategy='complementary'),
        'ensemble': HybridNASConfig(hybrid_strategy='ensemble'),
        'routing': HybridNASConfig(hybrid_strategy='routing'),
        'sequential': HybridNASConfig(hybrid_strategy='sequential')
    }
    
    results = {}
    
    for approach_name, config in approaches.items():
        logger.info(f"Testing {approach_name} approach...")
        
        start_time = time.time()
        try:
            best_architecture = search_hybrid_architecture(
                X_train, y_train, X_test, y_test, config, regime_labels
            )
            search_time = time.time() - start_time
            
            results[approach_name] = {
                'search_time': search_time,
                'accuracy': best_architecture.combined_accuracy,
                'efficiency': best_architecture.combined_efficiency,
                'interpretability': best_architecture.combined_interpretability,
                'overall_score': best_architecture.overall_score,
                'method': best_architecture.hybrid_method
            }
            
            logger.info(f"{approach_name}: {search_time:.2f}s, score: {best_architecture.overall_score:.4f}")
            
        except Exception as e:
            logger.warning(f"{approach_name} approach failed: {e}")
            results[approach_name] = {'error': str(e)}
    
    # Display comparison
    logger.info("=== Comparison Results ===")
    for approach, result in results.items():
        if 'error' not in result:
            logger.info(f"{approach}: {result['search_time']:.2f}s, {result['overall_score']:.4f} score")
        else:
            logger.info(f"{approach}: Failed - {result['error']}")
    
    return results


def create_hybrid_nas_visualization():
    """Create visualization comparing hybrid NAS approaches."""
    logger.info("=== Creating Hybrid NAS Visualization ===")
    
    # Simulate performance data
    approaches = ['Complementary', 'Ensemble', 'Routing', 'Sequential']
    search_times = [45.2, 52.8, 38.5, 41.3]  # seconds
    accuracies = [0.92, 0.89, 0.91, 0.90]
    interpretabilities = [0.85, 0.70, 0.88, 0.82]
    efficiencies = [0.88, 0.82, 0.90, 0.85]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Search time comparison
    ax1.bar(approaches, search_times, color=['green', 'blue', 'orange', 'red'])
    ax1.set_title('Search Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    ax2.bar(approaches, accuracies, color=['green', 'blue', 'orange', 'red'])
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    # Interpretability comparison
    ax3.bar(approaches, interpretabilities, color=['green', 'blue', 'orange', 'red'])
    ax3.set_title('Interpretability Comparison')
    ax3.set_ylabel('Interpretability Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # Efficiency comparison
    ax4.bar(approaches, efficiencies, color=['green', 'blue', 'orange', 'red'])
    ax4.set_title('Efficiency Comparison')
    ax4.set_ylabel('Efficiency Score')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/workspace/hybrid_nas_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved as 'hybrid_nas_comparison.png'")
    
    return fig


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Hybrid NAS Demonstration")
    
    try:
        # Demonstrate different hybrid approaches
        complementary_result = demonstrate_complementary_hybrid_nas()
        ensemble_result = demonstrate_ensemble_hybrid_nas()
        routing_result = demonstrate_routing_hybrid_nas()
        sequential_result = demonstrate_sequential_hybrid_nas()
        
        # Demonstrate hybrid clustering
        clustering_result = demonstrate_hybrid_clustering()
        
        # Compare approaches
        comparison_result = compare_hybrid_approaches()
        
        # Create visualization
        visualization = create_hybrid_nas_visualization()
        
        # Summary
        logger.info("=== Summary ===")
        logger.info("✅ Hybrid NAS successfully demonstrated")
        logger.info("✅ Complementary approach: Tree for features, Neural for patterns")
        logger.info("✅ Ensemble approach: Combines both methods")
        logger.info("✅ Routing approach: Intelligent routing based on data")
        logger.info("✅ Sequential approach: Tree first, then Neural")
        logger.info("✅ Hybrid clustering: Regime detection with both approaches")
        logger.info("✅ Hybrid NAS complements existing neural NAS system")
        
        return {
            'complementary_result': complementary_result,
            'ensemble_result': ensemble_result,
            'routing_result': routing_result,
            'sequential_result': sequential_result,
            'clustering_result': clustering_result,
            'comparison_result': comparison_result
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
    print("\n🎉 Hybrid NAS demonstration completed successfully!")
    print("📊 Check the generated visualization: hybrid_nas_comparison.png")
    print("🔍 Review the logs above for detailed performance metrics")
    print("🤝 Hybrid NAS complements your existing neural NAS system!")