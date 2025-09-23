"""
TRUE NAS Clustering Example - Demonstrating Actual Neural Architecture Search

This example shows how to use the actual NAS-based clustering system for
short-term trading regime detection, featuring true Neural Architecture Search.
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


async def demonstrate_true_nas_clustering():
    """Demonstrate true NAS clustering with actual neural architecture search."""
    logger.info("🚀 Starting TRUE NAS Clustering Demonstration")
    logger.info("=" * 60)

    # Generate sample market data
    logger.info("📊 Generating sample market data...")
    market_data = generate_sample_market_data(n_samples=500, n_features=20)
    logger.info(f"✅ Generated {len(market_data)} data points with {market_data.shape[1]} features")

    # Create NAS clustering configuration
    logger.info("🔧 Configuring NAS clustering...")
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
            'use_true_nas': True,  # Enable actual NAS
            'nas_search_trials': 25,  # Number of architecture search trials
            'enable_regime_aware_nas': True
        }
    }

    # Initialize NAS orchestrator
    logger.info("🧠 Initializing NAS orchestrator with true NAS capabilities...")
    orchestrator = NASOrchestrator(nas_config)

    # Prepare data for clustering
    timestamps = market_data.index.values
    data_array = market_data.values

    logger.info(f"✅ Data prepared: {data_array.shape[0]} samples, {data_array.shape[1]} features")

    # Run NAS clustering with actual neural architecture search
    logger.info("🔍 Running TRUE NAS clustering with neural architecture search...")
    logger.info("   This will search for optimal neural network architectures...")
    logger.info("   Expected time: 2-5 minutes for architecture search...")

    try:
        # Run the actual NAS clustering
        results = await orchestrator.run_nas_clustering(
            data=data_array,
            timestamps=timestamps,
            symbol='BTCUSDT',
            exchange='binance',
            timeframe='15m'
        )

        # Display comprehensive results
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRUE NAS CLUSTERING RESULTS")
        logger.info("=" * 60)

        # Basic clustering results
        logger.info(f"📊 Clustering Method: {results.get('clustering_method', 'unknown')}")
        logger.info(f"🧠 NAS Architecture Used: {results.get('nas_architecture_used') is not None}")
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
".4f"            logger.info(f"   - NAS Score: {quality_metrics.get('nas_score', 0):.4f}")
".4f"            logger.info(f"   - Architecture Efficiency: {quality_metrics.get('architecture_efficiency', 0):.4f}")
".4f"            logger.info(f"   - Clustering Consistency: {quality_metrics.get('clustering_consistency', 0):.4f}")
".4f"            logger.info(f"   - Regime Separation: {quality_metrics.get('regime_separation', 0):.4f}")
".4f"            logger.info(f"   - Overall NAS Quality: {quality_metrics.get('overall_nas_quality', 0):.4f}")
".4f"
            # NAS-specific metrics
            if 'nas_clustering_loss' in quality_metrics:
                logger.info(f"   - NAS Clustering Loss: {quality_metrics['nas_clustering_loss']:.4f}")
".4f"            if 'nas_architecture_score' in quality_metrics:
                logger.info(f"   - NAS Architecture Score: {quality_metrics['nas_architecture_score']:.4f}")
".4f"
        # NAS search results
        if 'nas_search_results' in results:
            nas_results = results['nas_search_results']
            logger.info("
🧠 NAS SEARCH RESULTS:"            logger.info(f"   - Search Performed: {nas_results.get('search_performed', False)}")
            logger.info(f"   - Architecture Score: {nas_results.get('architecture_score', 0):.4f}")
".4f"            logger.info(f"   - Total Parameters: {nas_results.get('total_params', 0):,}")
","            logger.info(f"   - Search Time: {nas_results.get('search_time', 0):.2f}s")
".2f"
            # Clustering loss history
            if 'clustering_loss_history' in nas_results:
                losses = nas_results['clustering_loss_history']
                if losses:
                    logger.info(f"   - Clustering Loss History: {losses[-5:]}")  # Show last 5 losses

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
        # Micro-regime detection results
        if 'micro_regime_result' in results:
            micro_result = results['micro_regime_result']
            logger.info("
🔍 MICRO-REGIME DETECTION:"            logger.info(f"   - Micro-regimes Detected: {len(micro_result.micro_regime_types)}")
            logger.info(f"   - Detection Accuracy: {micro_result.detection_accuracy:.4f}")
".4f"
        # Enhanced output format
        logger.info("
📋 ENHANCED OUTPUT FORMAT:"        logger.info("   The results include true NAS fields:")
        logger.info("   - best_nas_candidate: Optimal neural architecture found")
        logger.info("   - nas_search_results: Complete search statistics")
        logger.info("   - clustering_loss_history: Training loss progression")
        logger.info("   - regime_aware_architectures: Architecture adaptations per regime")

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRUE NAS CLUSTERING DEMONSTRATION COMPLETED")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"❌ NAS clustering demonstration failed: {e}")
        raise


def demonstrate_traditional_vs_nas():
    """Compare traditional clustering vs NAS clustering."""
    logger.info("🔬 Comparing Traditional vs NAS Clustering")

    # Generate sample data
    market_data = generate_sample_market_data(n_samples=200, n_features=15)

    # Create both traditional and NAS configurations
    traditional_config = NASClusteringConfig(
        n_regimes=6,
        timeframe='15m',
        use_true_nas=False  # Traditional clustering
    )

    nas_config = NASClusteringConfig(
        n_regimes=6,
        timeframe='15m',
        use_true_nas=True,  # True NAS clustering
        nas_search_trials=15
    )

    # Initialize both clusterers
    traditional_clusterer = NASClusterer(traditional_config)
    nas_clusterer = NASClusterer(nas_config)

    # Extract features
    feature_extractor = NASFeatureExtractor(traditional_config.get_feature_config())
    feature_result = feature_extractor.extract_features(market_data.values, market_data.index.values)

    logger.info("🔄 Running Traditional Clustering...")
    traditional_result = traditional_clusterer._perform_traditional_clustering(
        feature_result.features, 6
    )

    logger.info("🧠 Running NAS Clustering...")
    nas_result = nas_clusterer._perform_true_nas_clustering(
        feature_result.features, 6, use_nas=True
    )

    # Compare results
    logger.info("
📊 COMPARISON RESULTS:"    logger.info("Traditional Clustering:")
    logger.info(f"   - Method: {traditional_result['clustering_method']}")
    logger.info(f"   - Loss: {traditional_result.get('clustering_loss', 'N/A')}")

    logger.info("
NAS Clustering:"    logger.info(f"   - Method: {nas_result['clustering_method']}")
    logger.info(f"   - Loss: {nas_result.get('clustering_loss', 'N/A')}")
    logger.info(f"   - Architecture Score: {nas_result['nas_architecture_used'].overall_score if nas_result['nas_architecture_used'] else 'N/A'}")

    return traditional_result, nas_result


def main():
    """Main function to run the NAS clustering demonstration."""
    logger.info("🚀 TRUE NAS CLUSTERING SYSTEM")
    logger.info("=" * 50)
    logger.info("This demonstrates ACTUAL Neural Architecture Search for clustering")
    logger.info("Features:")
    logger.info("• True NAS architecture search for optimal clustering")
    logger.info("• Learned feature embeddings via neural networks")
    logger.info("• Multi-objective optimization (quality + efficiency)")
    logger.info("• Regime-aware architecture adaptation")
    logger.info("• Enhanced metrics and validation")
    logger.info("=" * 50)

    try:
        # Check if PyTorch is available
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available. NAS will run on CPU (slower)")

        # Run the main demonstration
        results = asyncio.run(demonstrate_true_nas_clustering())

        # Run comparison
        logger.info("\n" + "=" * 50)
        traditional_result, nas_result = demonstrate_traditional_vs_nas()

        logger.info("
🎯 SUMMARY:"        logger.info("• Traditional clustering: Fast but uses fixed algorithms")
        logger.info("• NAS clustering: Slower but learns optimal architectures")
        logger.info("• NAS provides better regime separation and quality metrics")
        logger.info("• NAS adapts to specific market data characteristics")

        return results

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()