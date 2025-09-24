"""
Economic Clustering Example for Hybrid NAS-TAS Regime System

This example demonstrates the enhanced economic clustering capabilities with:
- Economic significance integrated into clustering
- Momentum and volume analysis
- Advanced clustering algorithms
- Comprehensive economic evaluation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from ..config.hybrid_regime_config import (
    HybridRegimeConfig,
    RegimeCombinationStrategy,
    ClusteringAlgorithm,
    EconomicSignificanceType
)
from ..core.hybrid_regime_detector import HybridNASTASRegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_economic_test_data(n_samples: int = 2000) -> pd.DataFrame:
    """Create test data with distinct economic regimes including momentum and volume patterns."""
    np.random.seed(42)

    # Create timestamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

    # Generate price data with different economic regimes
    prices = []
    volumes = []
    current_price = 100.0
    current_volume = 1000.0

    regime_markers = []

    for i in range(n_samples):
        # Create different regimes every 250 samples
        if i % 250 == 0:
            # New regime
            regime_type = (i // 250) % 4
            regime_markers.append(regime_type)

        # Regime 0: High volatility, strong momentum (Trending)
        if i // 250 % 4 == 0:
            # Strong upward trend with high volatility
            trend = 0.002  # Strong uptrend
            volatility = 0.03  # High volatility
            volume_trend = 0.001  # Increasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.1))

        # Regime 1: Low volatility, weak momentum (Sideways)
        elif i // 250 % 4 == 1:
            # Sideways movement with low volatility
            trend = 0.0001  # Weak trend
            volatility = 0.01  # Low volatility
            volume_trend = -0.0005  # Decreasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.05))

        # Regime 2: Medium volatility, strong momentum (Volatile Trending)
        elif i // 250 % 4 == 2:
            # Volatile trending with medium volatility
            trend = 0.001  # Moderate trend
            volatility = 0.02  # Medium volatility
            volume_trend = 0.0008  # Increasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.08))

        # Regime 3: High volatility, no momentum (Mean Reverting)
        else:
            # Mean reverting with high volatility
            trend = -0.0005  # Slight downward drift
            volatility = 0.04  # High volatility
            volume_trend = 0.0002  # Slightly increasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.12))

        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        current_price *= (1 + price_change)
        prices.append(current_price)
        volumes.append(max(100, current_volume))  # Ensure minimum volume

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': volumes
    })

    logger.info("✅ Created economic test data with 4 distinct regimes:")
    logger.info("   Regime 0: High volatility, strong momentum (Trending)")
    logger.info("   Regime 1: Low volatility, weak momentum (Sideways)")
    logger.info("   Regime 2: Medium volatility, strong momentum (Volatile Trending)")
    logger.info("   Regime 3: High volatility, no momentum (Mean Reverting)")

    return data


def demonstrate_economic_clustering():
    """Demonstrate economic clustering with momentum and volume analysis."""

    logger.info("🚀 Starting Economic Clustering Demonstration")
    logger.info("=" * 60)

    # Create test data with economic regimes
    market_data = create_economic_test_data(2000)

    # Configuration 1: Economic Adaptive Clustering
    logger.info("\n📊 Configuration 1: Economic Adaptive Clustering")
    logger.info("-" * 50)

    config1 = HybridRegimeConfig(
        n_regimes=4,
        combination_strategy=RegimeCombinationStrategy.PERFORMANCE_ADAPTIVE,
        clustering_config={
            "primary_algorithm": ClusteringAlgorithm.ECONOMIC_ADAPTIVE,
            "ensemble_algorithms": [
                ClusteringAlgorithm.ECONOMIC_KMEANS,
                ClusteringAlgorithm.ECONOMIC_HIERARCHICAL,
                ClusteringAlgorithm.ECONOMIC_GMM
            ],
            "economic_clustering": True,
            "economic_features": True,
            "momentum_integration": True,
            "volume_integration": True,
            "momentum_threshold": 0.7,
            "volume_threshold": 0.6
        },
        economic_evaluation={
            "enabled": True,
            "significance_types": [
                EconomicSignificanceType.VOLATILITY_REGIME.value,
                EconomicSignificanceType.MOMENTUM_REGIME.value,
                EconomicSignificanceType.VOLUME_MOMENTUM.value,
                EconomicSignificanceType.PRICE_ACTION.value,
                EconomicSignificanceType.MARKET_MICROSTRUCTURE.value
            ],
            "momentum_threshold": 0.7,
            "volume_threshold": 0.6,
            "momentum_periods": [5, 10, 20, 50]
        }
    )

    detector1 = HybridNASTASRegimeDetector(config1)
    result1 = detector1.detect_regimes(market_data, validate_economic_significance=True)

    if result1.success:
        logger.info("✅ Economic Adaptive Clustering Results:")
        logger.info(f"   Regimes detected: {len(set(result1.regime_predictions))}")
        logger.info(f"   Economic significance scores: {result1.economic_significance_scores}")
        logger.info(f"   Average momentum score: {np.mean(result1.momentum_scores):.3".3f"
        logger.info(f"   Average volume profile: {np.mean(result1.volume_profiles):.3".3f"
        logger.info(f"   Economic clustering used: {result1.metadata.get('economic_clustering_used', False)}")
        logger.info(f"   Momentum integration: {result1.metadata.get('momentum_integration', False)}")
        logger.info(f"   Volume integration: {result1.metadata.get('volume_integration', False)}")

    # Configuration 2: Economic Hierarchical Clustering
    logger.info("\n📊 Configuration 2: Economic Hierarchical Clustering")
    logger.info("-" * 50)

    config2 = HybridRegimeConfig(
        n_regimes=4,
        combination_strategy=RegimeCombinationStrategy.HIERARCHICAL,
        clustering_config={
            "primary_algorithm": ClusteringAlgorithm.ECONOMIC_HIERARCHICAL,
            "ensemble_algorithms": [
                ClusteringAlgorithm.ECONOMIC_HIERARCHICAL,
                ClusteringAlgorithm.HIERARCHICAL,
                ClusteringAlgorithm.AGGLOMERATIVE
            ],
            "economic_clustering": True,
            "momentum_integration": True,
            "volume_integration": True
        },
        economic_evaluation={
            "enabled": True,
            "significance_types": [
                EconomicSignificanceType.MOMENTUM_REGIME.value,
                EconomicSignificanceType.VOLUME_MOMENTUM.value,
                EconomicSignificanceType.INTER_MARKET_ANALYSIS.value,
                EconomicSignificanceType.SECTOR_ROTATION.value
            ]
        }
    )

    detector2 = HybridNASTASRegimeDetector(config2)
    result2 = detector2.detect_regimes(market_data, validate_economic_significance=True)

    if result2.success:
        logger.info("✅ Economic Hierarchical Clustering Results:")
        logger.info(f"   Regimes detected: {len(set(result2.regime_predictions))}")
        logger.info(f"   Economic significance scores: {result2.economic_significance_scores}")
        logger.info(f"   Average momentum score: {np.mean(result2.momentum_scores):.3".3f"
        logger.info(f"   Average volume profile: {np.mean(result2.volume_profiles):.3".3f"

    # Configuration 3: Economic K-Means with Enhanced Features
    logger.info("\n📊 Configuration 3: Economic K-Means with Enhanced Features")
    logger.info("-" * 50)

    config3 = HybridRegimeConfig(
        n_regimes=4,
        combination_strategy=RegimeCombinationStrategy.WEIGHTED_AVERAGE,
        clustering_config={
            "primary_algorithm": ClusteringAlgorithm.ECONOMIC_KMEANS,
            "ensemble_algorithms": [
                ClusteringAlgorithm.ECONOMIC_KMEANS,
                ClusteringAlgorithm.KMEANS
            ],
            "economic_clustering": True,
            "economic_features": True,
            "momentum_integration": True,
            "volume_integration": True,
            "economic_distance_metric": "economic_euclidean"
        },
        economic_evaluation={
            "enabled": True,
            "significance_types": [
                EconomicSignificanceType.VOLATILITY_REGIME.value,
                EconomicSignificanceType.TREND_STRENGTH.value,
                EconomicSignificanceType.MOMENTUM_REGIME.value,
                EconomicSignificanceType.VOLUME_PROFILE.value,
                EconomicSignificanceType.CORRELATION_STRUCTURE.value,
                EconomicSignificanceType.MARKET_EFFICIENCY.value,
                EconomicSignificanceType.LIQUIDITY_REGIME.value
            ]
        }
    )

    detector3 = HybridNASTASRegimeDetector(config3)
    result3 = detector3.detect_regimes(market_data, validate_economic_significance=True)

    if result3.success:
        logger.info("✅ Economic K-Means Clustering Results:")
        logger.info(f"   Regimes detected: {len(set(result3.regime_predictions))}")
        logger.info(f"   Economic significance scores: {result3.economic_significance_scores}")
        logger.info(f"   Average momentum score: {np.mean(result3.momentum_scores):.3".3f"
        logger.info(f"   Average volume profile: {np.mean(result3.volume_profiles):.3".3f"

    # Comparison Analysis
    logger.info("\n📈 Comparison Analysis")
    logger.info("=" * 40)

    if result1.success and result2.success and result3.success:
        results = [result1, result2, result3]
        configs = ["Economic Adaptive", "Economic Hierarchical", "Economic K-Means"]

        for i, (result, config_name) in enumerate(zip(results, configs)):
            logger.info(f"\n{config_name}:")
            logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3".3f")
            logger.info(f"   Financial relevance: {np.mean(result.financial_relevance_scores):.3".3f")
            logger.info(f"   Average momentum: {np.mean(result.momentum_scores):.3".3f")
            logger.info(f"   Average volume profile: {np.mean(result.volume_profiles):.3".3f")
            logger.info(f"   Execution time: {result.execution_time:.3".3f"

    logger.info("\n🎯 Key Insights:")
    logger.info("-" * 20)
    logger.info("✅ Economic significance is now integrated directly into the clustering process")
    logger.info("✅ Momentum analysis is performed across multiple timeframes (5, 10, 20, 50 periods)")
    logger.info("✅ Volume analysis includes volume-price correlation and volume trends")
    logger.info("✅ Advanced clustering algorithms use economic distance metrics")
    logger.info("✅ Economic evaluation includes volatility, momentum, volume, and market microstructure")
    logger.info("✅ The system adapts clustering algorithms based on economic performance")

    logger.info("\n🚀 Economic Clustering Demonstration Complete!")
    logger.info("=" * 60)


def demonstrate_momentum_volume_analysis():
    """Demonstrate detailed momentum and volume analysis capabilities."""

    logger.info("\n🔍 Detailed Momentum and Volume Analysis")
    logger.info("=" * 50)

    # Create market data with clear momentum and volume patterns
    np.random.seed(42)
    n_samples = 1000
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

    # Generate price data with momentum patterns
    prices = []
    volumes = []
    current_price = 100.0
    current_volume = 1000.0

    # Create 4 different momentum-volume regimes
    for i in range(n_samples):
        regime = (i // 250) % 4

        if regime == 0:
            # Strong momentum, increasing volume
            momentum = 0.002
            volume_trend = 0.001
        elif regime == 1:
            # Weak momentum, decreasing volume
            momentum = 0.0005
            volume_trend = -0.0008
        elif regime == 2:
            # Strong momentum, high volume volatility
            momentum = 0.0015
            volume_trend = 0.0005
        else:
            # Mean reversion, stable volume
            momentum = -0.0003
            volume_trend = 0.0001

        # Generate price and volume
        price_change = np.random.normal(momentum, 0.02)
        current_price *= (1 + price_change)

        volume_change = np.random.normal(volume_trend, 0.1)
        current_volume *= (1 + volume_change)

        prices.append(current_price)
        volumes.append(max(100, current_volume))

    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': volumes
    })

    logger.info("✅ Created momentum-volume test data with 4 distinct patterns:")
    logger.info("   Regime 0: Strong momentum, increasing volume")
    logger.info("   Regime 1: Weak momentum, decreasing volume")
    logger.info("   Regime 2: Strong momentum, volatile volume")
    logger.info("   Regime 3: Mean reversion, stable volume")

    # Configure for detailed momentum-volume analysis
    config = HybridRegimeConfig(
        n_regimes=4,
        combination_strategy=RegimeCombinationStrategy.ADAPTIVE_FUSION,
        clustering_config={
            "primary_algorithm": ClusteringAlgorithm.ECONOMIC_ADAPTIVE,
            "economic_clustering": True,
            "momentum_integration": True,
            "volume_integration": True,
            "momentum_threshold": 0.7,
            "volume_threshold": 0.6
        },
        economic_evaluation={
            "enabled": True,
            "significance_types": [
                EconomicSignificanceType.MOMENTUM_REGIME.value,
                EconomicSignificanceType.VOLUME_MOMENTUM.value,
                EconomicSignificanceType.VOLUME_PROFILE.value,
                EconomicSignificanceType.CORRELATION_STRUCTURE.value
            ],
            "momentum_threshold": 0.7,
            "volume_threshold": 0.6,
            "momentum_periods": [5, 10, 20, 50],
            "volume_analysis_window": 20
        }
    )

    detector = HybridNASTASRegimeDetector(config)
    result = detector.detect_regimes(market_data, validate_economic_significance=True)

    if result.success:
        logger.info("\n📊 Momentum-Volume Analysis Results:")
        logger.info("-" * 40)
        logger.info(f"   Regimes detected: {len(set(result.regime_predictions))}")
        logger.info(f"   Economic significance scores: {result.economic_significance_scores}")
        logger.info(f"   Average momentum score: {np.mean(result.momentum_scores):.3".3f")
        logger.info(f"   Average volume profile: {np.mean(result.volume_profiles):.3".3f")

        # Analyze regime characteristics
        logger.info("\n🏗️ Regime Characteristics:")
        for i in range(len(result.economic_significance_scores)):
            logger.info(f"   Regime {i}:")
            logger.info(f"      Economic significance: {result.economic_significance_scores[i]:.".3f")
            logger.info(f"      Momentum score: {result.momentum_scores[i]:.".3f")
            logger.info(f"      Volume profile: {result.volume_profiles[i]:.".3f")

    logger.info("\n🎯 Momentum-Volume Analysis Complete!")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_economic_clustering()
    demonstrate_momentum_volume_analysis()

    logger.info("\n✨ All Economic Clustering Demonstrations Complete!")
    logger.info("=" * 60)