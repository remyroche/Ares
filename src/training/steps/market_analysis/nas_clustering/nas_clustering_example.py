"""
TRANSITIONAL NAS Clustering Example - Bridging Traditional + Advanced Methods

This example shows how to use the Transitional NAS system that bridges
traditional clustering methods with advanced NAS techniques using existing
optimization utilities (grid utils, Pareto analysis).
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
import torch

# Import the true NAS clustering components
from src.training.steps.market_analysis.nas_clustering import NASOrchestrator
from src.training.steps.market_analysis.nas_clustering.core.nas_clusterer import NASClusterer
from src.training.steps.market_analysis.nas_clustering.core.nas_config import NASClusteringConfig
from src.training.steps.market_analysis.nas_clustering.core.nas_feature_extractor import NASFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Generate sample market data for testing NAS clustering."""
    np.random.seed(42)

    # Generate timestamps (15-minute intervals)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15 * n_samples)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')

    # Generate realistic market data with different regimes
    data = []

    for i in range(n_samples):
        # Simulate different market regimes
        regime = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])  # 4 regimes

        if regime == 0:  # Bull market
            base_price = 50000 + np.random.normal(0, 100)
            volume = np.random.normal(1000, 200)
        elif regime == 1:  # Bear market
            base_price = 48000 + np.random.normal(0, 150)
            volume = np.random.normal(800, 150)
        elif regime == 2:  # High volatility
            base_price = 49000 + np.random.normal(0, 300)
            volume = np.random.normal(1500, 400)
        else:  # Low volatility consolidation
            base_price = 48500 + np.random.normal(0, 50)
            volume = np.random.normal(500, 100)

        # Generate OHLCV data
        open_price = base_price + np.random.normal(0, 20)
        high_price = max(open_price, base_price + abs(np.random.normal(0, 30)))
        low_price = min(open_price, base_price - abs(np.random.normal(0, 30)))
        close_price = base_price + np.random.normal(0, 15)

        # Add technical indicators as features
        features = []

        # Price-based features
        features.extend([open_price, high_price, low_price, close_price])

        # Volume features
        features.append(volume)

        # Technical indicators (simplified)
        if i > 20:  # Need some history for indicators
            # Moving averages
            ma_10 = np.mean([close_price for _ in range(min(10, i))])
            ma_20 = np.mean([close_price for _ in range(min(20, i))])
            features.extend([ma_10, ma_20])

            # RSI-like indicator
            gains = sum(1 for x in [close_price - close_price for _ in range(min(14, i))] if x > 0)
            rsi = (gains / min(14, i)) * 100 if i > 0 else 50
            features.append(rsi)

            # Volatility
            volatility = np.std([close_price for _ in range(min(20, i))])
            features.append(volatility)

        else:
            # Fill with zeros for initial data points
            features.extend([close_price] * 5)  # 5 technical indicators

        data.append(features)

    # Create DataFrame
    columns = ['open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_20', 'rsi', 'volatility'] + \
              [f'feature_{i}' for i in range(n_features - 9)]

    df = pd.DataFrame(data, columns=columns, index=timestamps)

    return df


async def demonstrate_transitional_nas_clustering():
    """Demonstrate Transitional NAS clustering bridging traditional + advanced methods."""
    logger.info("🚀 Starting TRANSITIONAL NAS Clustering Demonstration")
    logger.info("=" * 60)

    # Generate sample market data
    logger.info("📊 Generating sample market data...")
    market_data = generate_sample_market_data(n_samples=500, n_features=20)
    logger.info(f"✅ Generated {len(market_data)} data points with {market_data.shape[1]} features")

    # Create Transitional NAS clustering configuration
    logger.info("🔧 Configuring Transitional NAS clustering...")
    nas_config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data',
        'nas_config': {
            'n_regimes': 8,
            'enable_micro_regime_detection': True,
            'economic_significance_threshold': 0.7,
            'trading_viability_threshold': 0.6,

            # Transitional NAS settings
            'enable_transitional_nas': True,
            'search_strategy': 'transitional',  # Uses grid + TPE + Pareto

            # Grid-optimized TPE using existing utilities
            'tpe_optimization': True,
            'use_grid_for_tpe': True,
            'grid_coarse_points': 8,
            'grid_fine_points': 5,

            # Pareto analysis using existing utilities
            'pareto_optimization': True,
            'pareto_objectives': ['clustering_quality', 'efficiency', 'robustness'],

            # Focus on essential methods only
            'clustering_methods': ['kmeans', 'neural'],
            'fusion_strategy': 'hybrid'
        }
    }

    # Initialize NAS orchestrator
    logger.info("🧠 Initializing Transitional NAS orchestrator...")
    orchestrator = NASOrchestrator(nas_config)

    # Prepare data for clustering
    timestamps = market_data.index.values
    data_array = market_data.values

    logger.info(f"✅ Data prepared: {data_array.shape[0]} samples, {data_array.shape[1]} features")

    # Run Transitional NAS clustering
    logger.info("🔍 Running Transitional NAS clustering...")
    logger.info("   Phase 1: Grid-optimized TPE search using existing utilities")
    logger.info("   Phase 2: Pareto front analysis for optimal trade-offs")
    logger.info("   Phase 3: Hybrid fusion (K-means + Neural embeddings)")
    logger.info("   Expected time: 1-3 minutes using existing optimization utilities...")

    try:
        # Run the Transitional NAS clustering
        results = await orchestrator.run_nas_clustering(
            data=data_array,
            timestamps=timestamps,
            symbol='BTCUSDT',
            exchange='binance',
            timeframe='15m'
        )

        # Display comprehensive results
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRANSITIONAL NAS CLUSTERING RESULTS")
        logger.info("=" * 60)

        # Basic clustering results
        logger.info(f"📊 Clustering Method: {results.get('clustering_method', 'unknown')}")
        logger.info(f"🔍 Search Strategy: {results.get('search_strategy', 'transitional')}")
        logger.info(f"⏱️  Execution Time: {results.get('execution_time', 0):.2f}s")
        logger.info(f"✅ Success: {results.get('success', False)}")

        # Clustering statistics
        if 'clustering_result' in results:
            clustering_result = results['clustering_result']
            logger.info("
📈 CLUSTERING STATISTICS:"            logger.info(f"   - Number of Regimes: {len(np.unique(clustering_result.labels))}")
            logger.info(f"   - Regime Distribution: {np.bincount(clustering_result.labels)}")

            # Quality metrics
            quality_metrics = clustering_result.quality_metrics
            logger.info("
🔬 QUALITY METRICS:"            logger.info(f"   - Silhouette Score: {quality_metrics.get('silhouette_score', 0):.4f}")
".4f"            logger.info(f"   - Calinski-Harabasz Score: {quality_metrics.get('calinski_harabasz_score', 0):.4f}")
".4f"            logger.info(f"   - Transitional Score: {quality_metrics.get('transitional_score', 0):.4f}")
".4f"            logger.info(f"   - Grid-TPE Efficiency: {quality_metrics.get('grid_tpe_efficiency', 0):.4f}")
".4f"            logger.info(f"   - Pareto Quality: {quality_metrics.get('pareto_quality', 0):.4f}")
".4f"
        # Transitional NAS results
        if 'transitional_results' in results:
            transitional_results = results['transitional_results']
            logger.info("
🔄 TRANSITIONAL NAS RESULTS:"            logger.info(f"   - K-means Quality: {transitional_results.get('kmeans_quality', 0):.4f}")
".4f"            logger.info(f"   - Neural Quality: {transitional_results.get('neural_quality', 0):.4f}")
".4f"            logger.info(f"   - Hybrid Improvement: {transitional_results.get('improvement_score', 0):.4f}")
".4f"            logger.info(f"   - Fusion Type: {transitional_results.get('fusion_type', 'hybrid')}")

        # Economic significance and trading viability
        if 'economic_significance_scores' in results:
            economic_scores = results['economic_significance_scores']
            trading_scores = results['trading_viability_scores']

            logger.info("
💰 ECONOMIC ANALYSIS:"            logger.info(f"   - Avg Economic Significance: {np.mean(economic_scores):.4f}")
".4f"            logger.info(f"   - Avg Trading Viability: {np.mean(trading_scores):.4f}")
".4f"
            # Regime analysis
            unique_regimes = np.unique(results['clustering_result'].labels)
            for regime in unique_regimes:
                regime_mask = results['clustering_result'].labels == regime
                regime_economic = np.mean(economic_scores[regime_mask])
                regime_trading = np.mean(trading_scores[regime_mask])
                regime_count = np.sum(regime_mask)

                logger.info(f"   - Regime {regime}: {regime_count} samples, "
                           f"Economic: {regime_economic:.4f}, ".4f"                           f"Trading: {regime_trading:.4f}")
".4f"
        # Key advantages of Transitional NAS
        logger.info("
✅ TRANSITIONAL NAS ADVANTAGES:"        logger.info("   ✓ Uses existing grid utilities for TPE optimization")
        logger.info("   ✓ Uses existing Pareto utilities for multi-objective analysis")
        logger.info("   ✓ Focuses on essential methods (K-means + Neural embeddings)")
        logger.info("   ✓ Simpler to understand and debug")
        logger.info("   ✓ Production-ready using proven optimization infrastructure")

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRANSITIONAL NAS CLUSTERING DEMONSTRATION COMPLETED")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"❌ Transitional NAS clustering demonstration failed: {e}")
        raise


def demonstrate_transitional_vs_traditional():
    """Compare traditional clustering vs Transitional NAS clustering."""
    logger.info("🔬 Comparing Traditional vs Transitional NAS Clustering")

    # Generate sample data
    market_data = generate_sample_market_data(n_samples=200, n_features=15)

    # Create configurations
    traditional_config = NASClusteringConfig(
        n_regimes=6,
        timeframe='15m',
        search_strategy='standard'  # Traditional clustering
    )

    transitional_config = NASClusteringConfig(
        n_regimes=6,
        timeframe='15m',
        search_strategy='transitional',  # Transitional NAS
        enable_transitional_nas=True,
        tpe_optimization=True,
        pareto_optimization=True,
        clustering_methods=['kmeans', 'neural']  # Essential methods only
    )

    # Initialize clusterers
    traditional_clusterer = NASClusterer(traditional_config)
    transitional_clusterer = NASClusterer(transitional_config)

    # Extract features
    feature_extractor = NASFeatureExtractor(traditional_config.get_feature_config())
    feature_result = feature_extractor.extract_features(market_data.values, market_data.index.values)

    logger.info("🔄 Running Traditional Clustering...")
    traditional_result = traditional_clusterer._perform_traditional_clustering(
        feature_result.features, 6
    )

    logger.info("🧠 Running Transitional NAS Clustering...")
    transitional_result = transitional_clusterer._perform_transitional_nas_search(
        feature_result.features, 6
    )

    # Compare results
    logger.info("
📊 COMPARISON RESULTS:"    logger.info("Traditional Clustering:")
    logger.info(f"   - Method: {traditional_result['clustering_method']}")
    logger.info(f"   - Quality: {traditional_result.get('quality', 'N/A')}")

    logger.info("
Transitional NAS Clustering:"    logger.info(f"   - Method: {transitional_result.get('method', 'transitional')}")
    logger.info(f"   - K-means Quality: {transitional_result.get('kmeans_quality', 'N/A')}")
    logger.info(f"   - Neural Quality: {transitional_result.get('neural_quality', 'N/A')}")
    logger.info(f"   - Hybrid Improvement: {transitional_result.get('improvement_score', 'N/A')}")

    # Calculate improvement
    if 'improvement_score' in transitional_result:
        improvement = transitional_result['improvement_score']
        logger.info(f"   - Overall Improvement: {improvement:.4f} ({improvement*100:.1f}%".1f"        if improvement > 0:
            logger.info("   - ✅ Transitional NAS shows improvement!")
        else:
            logger.info("   - ⚠️  Transitional NAS needs tuning")

    return traditional_result, transitional_result


def main():
    """Main function to run the Transitional NAS clustering demonstration."""
    logger.info("🚀 TRANSITIONAL NAS CLUSTERING SYSTEM")
    logger.info("=" * 50)
    logger.info("This demonstrates TRANSITIONAL Neural Architecture Search for clustering")
    logger.info("Features:")
    logger.info("• Grid-optimized TPE using existing utilities")
    logger.info("• Pareto front analysis for optimal trade-offs")
    logger.info("• Essential methods only (K-means + Neural embeddings)")
    logger.info("• Hybrid fusion bridging traditional + advanced approaches")
    logger.info("• Production-ready using proven optimization infrastructure")
    logger.info("=" * 50)

    try:
        # Check if PyTorch is available
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available. NAS will run on CPU (slower)")

        # Run the main demonstration
        results = asyncio.run(demonstrate_transitional_nas_clustering())

        # Run comparison
        logger.info("\n" + "=" * 50)
        traditional_result, transitional_result = demonstrate_transitional_vs_traditional()

        logger.info("
🎯 SUMMARY:"        logger.info("• Traditional clustering: Fast but uses fixed algorithms")
        logger.info("• Transitional NAS: Uses existing utilities (grid + Pareto)")
        logger.info("• Focuses on essential methods (K-means + Neural embeddings)")
        logger.info("• Hybrid approach bridges traditional and advanced methods")
        logger.info("• Production-ready using proven optimization infrastructure")

        # Key benefits
        if 'transitional_results' in results:
            improvement = results['transitional_results'].get('improvement_score', 0)
            logger.info(f"• Performance improvement: {improvement:.4f} ({improvement*100:.1f}%".1f"        if improvement > 0:
            logger.info("• ✅ Transitional NAS successfully improves clustering quality!")

        return results

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()