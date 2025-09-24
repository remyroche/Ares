"""
Tree-Based Clustering Regime Detection Example

This example demonstrates the new tree-based clustering capabilities in TAS,
including data-driven strategy selection, clustering metrics, and market-specific optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

from ..core.tas_config import TASConfig, TreeModelType, ClusteringStrategy, ClusteringMetric
from ..regime_analysis.clustering_regime_detection import (
    TreeBasedClusteringRegimeDetector,
    ClusteringRegimeConfig,
    quick_clustering_detection
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    # Create time index
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')

    # Generate OHLCV data with different market regimes
    data = []

    for i in range(n_samples):
        # Simulate different market conditions
        if i < n_samples // 3:
            # Trending market
            base_price = 100 + i * 0.1 + np.random.normal(0, 0.5)
            volume = 1000 + np.random.normal(0, 200)
        elif i < 2 * n_samples // 3:
            # Volatile market
            base_price = 130 + np.random.normal(0, 2.0)
            volume = 1500 + np.random.normal(0, 400)
        else:
            # Mean-reverting market
            base_price = 120 + np.sin(i * 0.1) * 5 + np.random.normal(0, 1.0)
            volume = 800 + np.random.normal(0, 150)

        # Create OHLCV
        open_price = base_price + np.random.normal(0, 0.2)
        high_price = max(open_price, base_price + abs(np.random.normal(0, 0.5)))
        low_price = min(open_price, base_price - abs(np.random.normal(0, 0.5)))
        close_price = base_price + np.random.normal(0, 0.2)
        volume = max(100, volume)

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data)


def example_basic_clustering():
    """Example 1: Basic tree-based clustering regime detection."""
    logger.info("=== Example 1: Basic Tree-Based Clustering ===")

    # Generate sample data
    market_data = generate_sample_market_data(500)
    logger.info(f"Generated market data: {len(market_data)} samples")

    # Configure clustering
    config = ClusteringRegimeConfig(
        clustering_strategy="auto",
        n_regimes=4,
        tabular_threshold=0.7,
        sequential_threshold=0.5,
        clustering_metrics=["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
    )

    # Create detector
    detector = TreeBasedClusteringRegimeDetector(config)

    # Detect regimes
    results = detector.detect_regimes(market_data)

    # Display results
    logger.info(f"Detected {len(set(results['labels']))} regimes")
    logger.info(f"Clustering strategy used: {results['strategy']}")
    logger.info(f"Clustering metrics: {results['clustering_metrics']}")
    logger.info(f"Data characteristics: {results['data_characteristics']}")

    return results


def example_advanced_clustering():
    """Example 2: Advanced clustering with specific strategies."""
    logger.info("=== Example 2: Advanced Tree-Based Clustering ===")

    market_data = generate_sample_market_data(800)

    # Example with different strategies
    strategies = ["complementary", "ensemble", "sequential", "single"]

    for strategy in strategies:
        logger.info(f"\n--- Testing strategy: {strategy} ---")

        config = ClusteringRegimeConfig(
            clustering_strategy=strategy,
            n_regimes=5,
            enable_feature_selection=True,
            max_features_per_model=30
        )

        detector = TreeBasedClusteringRegimeDetector(config)
        results = detector.detect_regimes(market_data)

        logger.info(f"Strategy: {strategy}")
        logger.info(f"Method used: {results.get('method', 'unknown')}")
        logger.info(f"Clustering metrics: {results['clustering_metrics']}")
        logger.info(f"Number of regimes: {len(set(results['labels']))}")


def example_market_specific_analysis():
    """Example 3: Market-specific data analysis and strategy selection."""
    logger.info("=== Example 3: Market-Specific Analysis ===")

    # Generate data with specific market characteristics
    market_data = generate_sample_market_data(600)

    # Analyze market characteristics
    data_characteristics = {
        'n_samples': len(market_data),
        'n_features': len(market_data.columns),
        'tabular_ratio': 0.8,  # High tabular ratio -> complementary strategy
        'sequential_ratio': 0.3,
        'complexity_ratio': 0.4,
        'volatility': market_data['close'].pct_change().std(),
        'volume_ratio': market_data['volume'].mean() / market_data['volume'].std(),
        'is_tabular_dominant': True,
        'is_sequential_dominant': False,
        'is_complex_dominant': False,
        'is_volatile': False,
        'has_high_volume_ratio': False
    }

    logger.info(f"Market data characteristics: {data_characteristics}")

    # Create detector with auto strategy selection
    config = ClusteringRegimeConfig(
        clustering_strategy="auto",
        n_regimes=6,
        tabular_threshold=0.7,
        sequential_threshold=0.5,
        complexity_threshold=0.8
    )

    detector = TreeBasedClusteringRegimeDetector(config)
    results = detector.detect_regimes(market_data)

    logger.info(f"Auto-selected strategy: {results['strategy']}")
    logger.info(f"Strategy selection based on data characteristics:")
    logger.info(f"  - Tabular dominant: {data_characteristics['is_tabular_dominant']}")
    logger.info(f"  - Sequential dominant: {data_characteristics['is_sequential_dominant']}")
    logger.info(f"  - Complex dominant: {data_characteristics['is_complex_dominant']}")

    return results


def example_clustering_metrics_analysis():
    """Example 4: Detailed clustering metrics analysis."""
    logger.info("=== Example 4: Clustering Metrics Analysis ===")

    market_data = generate_sample_market_data(400)

    config = ClusteringRegimeConfig(
        clustering_strategy="ensemble",
        n_regimes=4,
        clustering_metrics=["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
    )

    detector = TreeBasedClusteringRegimeDetector(config)
    results = detector.detect_regimes(market_data)

    logger.info("Detailed Clustering Metrics Analysis:")
    metrics = results['clustering_metrics']

    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    # Interpret results
    silhouette = metrics.get('silhouette_score', 0)
    calinski = metrics.get('calinski_harabasz_score', 0)
    davies = metrics.get('davies_bouldin_score', 0)

    logger.info("\nClustering Quality Interpretation:")
    if silhouette > 0.5:
        logger.info("  ✓ Excellent clustering (silhouette > 0.5)")
    elif silhouette > 0.25:
        logger.info("  ✓ Good clustering (silhouette > 0.25)")
    elif silhouette > 0:
        logger.info("  ✓ Fair clustering (silhouette > 0)")
    else:
        logger.info("  ✗ Poor clustering (silhouette ≤ 0)")

    if calinski > 1000:
        logger.info("  ✓ Excellent separation (Calinski-Harabasz > 1000)")
    elif calinski > 100:
        logger.info("  ✓ Good separation (Calinski-Harabasz > 100)")

    if davies < 1:
        logger.info("  ✓ Excellent compactness (Davies-Bouldin < 1)")
    elif davies < 2:
        logger.info("  ✓ Good compactness (Davies-Bouldin < 2)")


def example_tas_integration():
    """Example 5: Integration with TAS for advanced tree models."""
    logger.info("=== Example 5: TAS Integration with New Tree Models ===")

    # Generate market data
    market_data = generate_sample_market_data(300)

    # Create TAS configuration with new tree models
    tas_config = TASConfig(
        enable_unsupervised_regime_detection=True,
        enable_data_driven_strategy_selection=True,
        clustering_strategy=ClusteringStrategy.ENSEMBLE,
        clustering_metrics=[
            ClusteringMetric.SILHOUETTE,
            ClusteringMetric.CALINSKI_HARABASZ,
            ClusteringMetric.DAVIES_BOULDIN
        ],
        model_types=[
            TreeModelType.RANDOM_FOREST,
            TreeModelType.XGBOOST,
            TreeModelType.LIGHTGBM,
            TreeModelType.NGBOOST,
            TreeModelType.DART,
            TreeModelType.QUANTILE_GBDT
        ]
    )

    logger.info(f"TAS configured with {len(tas_config.model_types)} tree models")
    logger.info("Available models:")
    for model in tas_config.model_types:
        logger.info(f"  - {model.value}")

    logger.info(f"Clustering strategy: {tas_config.clustering_strategy.value}")
    logger.info(f"Clustering metrics: {[m.value for m in tas_config.clustering_metrics]}")


def main():
    """Run all examples."""
    logger.info("🚀 Starting Tree-Based Clustering Examples")
    logger.info("=" * 60)

    try:
        # Run examples
        results1 = example_basic_clustering()
        example_advanced_clustering()
        results3 = example_market_specific_analysis()
        example_clustering_metrics_analysis()
        example_tas_integration()

        logger.info("=" * 60)
        logger.info("✅ All examples completed successfully!")

        # Summary
        logger.info("Summary of new TAS clustering capabilities:")
        logger.info("1. ✓ Data-driven clustering strategy selection")
        logger.info("2. ✓ Tree-based clustering strategies (no neural)")
        logger.info("3. ✓ Market data analysis for strategy selection")
        logger.info("4. ✓ Clustering-specific metrics and quality assessment")
        logger.info("5. ✓ Market-specific optimization features")
        logger.info("6. ✓ Support for advanced tree models (NGBoost, DART, etc.)")
        logger.info("7. ✓ Standalone unsupervised regime detection")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()