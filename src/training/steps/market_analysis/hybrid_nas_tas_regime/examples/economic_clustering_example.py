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


def create_economic_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data with distinct economic regimes for 15m short-term trading."""
    np.random.seed(42)

    # Create timestamps for 15m intervals
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')

    # Generate price data with different economic regimes optimized for 15m trading
    prices = []
    volumes = []
    current_price = 100.0
    current_volume = 1000.0

    regime_markers = []

    for i in range(n_samples):
        # Create different regimes every 100 samples (25 hours of 15m bars)
        if i % 100 == 0:
            # New regime
            regime_type = (i // 100) % 4
            regime_markers.append(regime_type)

        # Regime 0: Short-term momentum (15m-2.5h) with high volume
        if i // 100 % 4 == 0:
            # Strong short-term momentum (15m timeframe focus)
            trend = 0.0015  # Moderate short-term trend
            volatility = 0.02  # Medium volatility for 15m
            volume_trend = 0.002  # Increasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.15))

        # Regime 1: Low volatility, micro patterns (15m intra-bar focus)
        elif i // 100 % 4 == 1:
            # Low volatility with intra-bar patterns
            trend = 0.0002  # Very weak trend
            volatility = 0.008  # Low volatility for 15m
            volume_trend = -0.001  # Decreasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.08))

        # Regime 2: High volatility, rapid rotations (15m sector rotation focus)
        elif i // 100 % 4 == 2:
            # High volatility with rapid rotations (15m-1h)
            trend = 0.0008  # Moderate trend
            volatility = 0.025  # Higher volatility for 15m
            volume_trend = 0.0015  # Volatile volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.20))

        # Regime 3: Microstructure patterns (15m microstructure focus)
        else:
            # Microstructure patterns with mean reversion
            trend = -0.0003  # Slight mean reversion
            volatility = 0.015  # Medium volatility
            volume_trend = 0.0005  # Slightly increasing volume
            current_volume *= (1 + np.random.normal(volume_trend, 0.12))

        # Generate price movement optimized for 15m bars
        price_change = np.random.normal(trend, volatility)
        current_price *= (1 + price_change)
        prices.append(current_price)
        volumes.append(max(50, current_volume))  # Lower minimum volume for 15m

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],  # Tighter spreads for 15m
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],   # Tighter spreads for 15m
        'close': prices,
        'volume': volumes
    })

    logger.info("✅ Created economic test data optimized for 15m short-term trading:")
    logger.info("   Regime 0: Short-term momentum (15m-2.5h) with high volume")
    logger.info("   Regime 1: Low volatility, micro patterns (15m intra-bar focus)")
    logger.info("   Regime 2: High volatility, rapid rotations (15m sector rotation)")
    logger.info("   Regime 3: Microstructure patterns (15m microstructure focus)")
    logger.info("   Timeframe: 15m bars, optimized for short-term trading (5-30m)")

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
        logger.info(f"   Average momentum score: {np.mean(result1.momentum_scores):.3f}")
        logger.info(f"   Average volume profile: {np.mean(result1.volume_profiles):.3f}")
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
        logger.info(f"   Average momentum score: {np.mean(result2.momentum_scores):.3f}")
        logger.info(f"   Average volume profile: {np.mean(result2.volume_profiles):.3f}")

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
        logger.info(f"   Average momentum score: {np.mean(result3.momentum_scores):.3f}")
        logger.info(f"   Average volume profile: {np.mean(result3.volume_profiles):.3f}")

    # Comparison Analysis
    logger.info("\n📈 Comparison Analysis")
    logger.info("=" * 40)

    if result1.success and result2.success and result3.success:
        results = [result1, result2, result3]
        configs = ["Economic Adaptive", "Economic Hierarchical", "Economic K-Means"]

        for i, (result, config_name) in enumerate(zip(results, configs)):
            logger.info(f"\n{config_name}:")
            logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
            logger.info(f"   Financial relevance: {np.mean(result.financial_relevance_scores):.3f}")
            logger.info(f"   Average momentum: {np.mean(result.momentum_scores):.3f}")
            logger.info(f"   Average volume profile: {np.mean(result.volume_profiles):.3f}")
            logger.info(f"   Execution time: {result.execution_time:.3f}")

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
    """Demonstrate detailed momentum and volume analysis capabilities for 15m trading."""

    logger.info("\n🔍 Detailed Short-Term Momentum and Volume Analysis")
    logger.info("=" * 60)

    # Create market data with clear short-term momentum and volume patterns for 15m trading
    np.random.seed(42)
    n_samples = 800  # ~200 hours of 15m bars
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')

    # Generate price data with short-term momentum patterns (15m-2.5h focus)
    prices = []
    volumes = []
    current_price = 100.0
    current_volume = 1000.0

    # Create 4 different short-term momentum-volume regimes
    for i in range(n_samples):
        regime = (i // 200) % 4  # Shorter regimes for 15m analysis

        if regime == 0:
            # Short-term momentum (15m-1h), high volume participation
            momentum = 0.0012  # Moderate short-term momentum
            volume_trend = 0.002  # Strong volume participation
        elif regime == 1:
            # Micro momentum (15m-30m), low volume
            momentum = 0.0004  # Weak short-term momentum
            volume_trend = -0.001  # Low volume participation
        elif regime == 2:
            # Rapid rotations (15m-45m), volatile volume
            momentum = 0.0008  # Moderate momentum with rotations
            volume_trend = 0.0015  # High volume volatility
        else:
            # Microstructure patterns (15m), stable volume
            momentum = -0.0002  # Slight mean reversion
            volume_trend = 0.0003  # Slightly increasing volume

        # Generate price and volume optimized for 15m bars
        price_change = np.random.normal(momentum, 0.015)  # Lower volatility for 15m
        current_price *= (1 + price_change)

        volume_change = np.random.normal(volume_trend, 0.12)  # Volume volatility for 15m
        current_volume *= (1 + volume_change)

        prices.append(current_price)
        volumes.append(max(50, current_volume))  # Lower minimum for 15m

    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],  # Tighter spreads
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],   # Tighter spreads
        'close': prices,
        'volume': volumes
    })

    logger.info("✅ Created short-term momentum-volume test data optimized for 15m trading:")
    logger.info("   Timeframe: 15m bars (200 hours of data)")
    logger.info("   Regime 0: Short-term momentum (15m-1h) with high volume participation")
    logger.info("   Regime 1: Micro momentum (15m-30m) with low volume participation")
    logger.info("   Regime 2: Rapid rotations (15m-45m) with volatile volume")
    logger.info("   Regime 3: Microstructure patterns (15m) with stable volume")

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
        logger.info(f"   Average momentum score: {np.mean(result.momentum_scores):.3f}")
        logger.info(f"   Average volume profile: {np.mean(result.volume_profiles):.3f}")

        # Analyze regime characteristics
        logger.info("\n🏗️ Regime Characteristics:")
        for i in range(len(result.economic_significance_scores)):
            logger.info(f"   Regime {i}:")
            logger.info(f"      Economic significance: {result.economic_significance_scores[i]:.3f}")
            logger.info(f"      Momentum score: {result.momentum_scores[i]:.3f}")
            logger.info(f"      Volume profile: {result.volume_profiles[i]:.3f}")

    logger.info("\n🎯 Momentum-Volume Analysis Complete!")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_economic_clustering()
    demonstrate_momentum_volume_analysis()

    logger.info("\n✨ All Economic Clustering Demonstrations Complete!")
    logger.info("=" * 60)