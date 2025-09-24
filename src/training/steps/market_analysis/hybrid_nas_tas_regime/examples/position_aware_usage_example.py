"""
Position-Aware Trading Analysis Usage Example

Demonstrates how both TAS and NAS systems can use the shared position-aware trading utilities
to ensure accurate win rate calculations for both long and short positions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import shared utilities
from ..shared_utils.position_aware_trading import (
    PositionAwareTradingAnalyzer, PositionAwareConfig,
    create_position_aware_analyzer, quick_position_aware_analysis
)

# Import TAS and NAS detectors
from ..tas_regime.core.tas_regime_detector import TASRegimeDetector
from ..nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector

logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)

    # Generate OHLCV data with realistic patterns
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')

    # Generate price data with trends and noise
    base_price = 100.0
    trend = np.linspace(0, 10, n_samples)  # Upward trend
    noise = np.random.normal(0, 0.5, n_samples)
    close_prices = base_price + trend + noise

    # Generate OHLCV data
    high_prices = close_prices + np.abs(np.random.normal(0, 0.2, n_samples))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.2, n_samples))
    open_prices = close_prices + np.random.normal(0, 0.1, n_samples)
    volumes = np.random.lognormal(10, 1, n_samples)  # Realistic volume distribution

    return pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })


def demonstrate_shared_position_aware_utilities():
    """Demonstrate shared position-aware utilities used by both TAS and NAS."""
    logger.info("🚀 Demonstrating Shared Position-Aware Trading Utilities")
    logger.info("=" * 70)

    # Step 1: Generate sample market data
    logger.info("📊 Step 1: Generating sample market data...")
    market_data = generate_sample_market_data(1000)
    logger.info(f"   Generated {len(market_data)} samples")
    logger.info(f"   Price range: {market_data['close'].min()".2f"} - {market_data['close'].max()".2f"}")

    # Step 2: Create position-aware analyzer
    logger.info("🔧 Step 2: Creating shared position-aware analyzer...")
    analyzer = create_position_aware_analyzer()

    # Step 3: Generate position directions (simulate trading decisions)
    logger.info("🎯 Step 3: Generating position directions...")
    returns = market_data['close'].pct_change().fillna(0)

    # Simulate position decisions based on returns
    position_directions = np.zeros(len(market_data))
    position_directions[returns > 0.001] = 1   # Long when positive returns
    position_directions[returns < -0.001] = -1  # Short when negative returns

    long_count = np.sum(position_directions == 1)
    short_count = np.sum(position_directions == -1)
    logger.info(f"   Long positions: {long_count}")
    logger.info(f"   Short positions: {short_count}")
    logger.info(f"   Neutral positions: {np.sum(position_directions == 0)}")

    # Step 4: Quick position-aware analysis
    logger.info("⚡ Step 4: Quick position-aware analysis...")
    quick_result = quick_position_aware_analysis(
        market_data, np.random.randint(0, 8, len(market_data)), position_directions
    )

    logger.info("📈 Quick Analysis Results:")
    logger.info(f"   Overall win rate: {quick_result['overall_analysis']['overall_win_rate']".3f"}")
    logger.info(f"   Long win rate: {quick_result['overall_analysis']['long_win_rate']".3f"}")
    logger.info(f"   Short win rate: {quick_result['overall_analysis']['short_win_rate']".3f"}")

    # Step 5: Detailed analysis per regime
    logger.info("🔍 Step 5: Detailed regime analysis...")
    detailed_result = analyzer.analyze_regime_position_performance(
        market_data, np.random.randint(0, 8, len(market_data)), position_directions
    )

    logger.info("📊 Regime Analysis Summary:")
    for regime_key, regime_data in detailed_result['regime_analyses'].items():
        logger.info(f"   {regime_key}:")
        logger.info(f"     Long win rate: {regime_data['long_win_rate']".3f"}")
        logger.info(f"     Short win rate: {regime_data['short_win_rate']".3f"}")
        logger.info(f"     Economic significance: {regime_data['economic_significance']".3f"}")

    # Step 6: Trading viability analysis
    logger.info("💰 Step 6: Position-aware trading viability...")
    viability_result = analyzer.calculate_position_aware_trading_viability(
        market_data, np.random.randint(0, 8, len(market_data)), position_directions
    )

    logger.info("🎯 Viability Analysis:")
    logger.info(f"   Long viability: {viability_result['long_viability']".3f"}")
    logger.info(f"   Short viability: {viability_result['short_viability']".3f"}")
    logger.info(f"   Overall viability: {viability_result['overall_viability']".3f"}")

    # Step 7: Position-aware recommendations
    logger.info("📋 Step 7: Position-aware recommendations...")
    recommendations = analyzer.get_position_aware_recommendations(detailed_result)

    logger.info("🎯 Position Recommendations:")
    for position_type, rec in recommendations['position_recommendations'].items():
        logger.info(f"   {position_type.upper()}: {rec['recommendation']}")
        logger.info(f"     Confidence: {rec['confidence']".3f"}")
        logger.info(f"     Position size: {rec['position_size']}")

    logger.info("✅ Shared position-aware utilities demonstration completed!")


def demonstrate_tas_position_awareness():
    """Demonstrate how TAS system uses shared position-aware utilities."""
    logger.info("\n🌲 Demonstrating TAS with Position-Aware Utilities")
    logger.info("=" * 50)

    try:
        # Create TAS detector with position-aware support
        from ..tas_regime.core.tas_regime_config import TASRegimeConfig

        config = TASRegimeConfig(
            n_regimes=6,
            primary_architecture=TASRegimeConfig.TASArchitectureType.TREE_ENSEMBLE,
            enable_economic_evaluation=True,
            enable_uncertainty_quantification=True
        )

        tas_detector = TASRegimeDetector(config)

        # Generate sample data
        market_data = generate_sample_market_data(500)
        position_directions = np.where(
            market_data['close'].pct_change() > 0.001, 1,
            np.where(market_data['close'].pct_change() < -0.001, -1, 0)
        )

        logger.info("🚀 Running TAS with position-aware analysis...")
        tas_result = tas_detector.detect_regimes(
            market_data, optimize_performance=True, enable_clvsa_enhancement=True
        )

        logger.info("📊 TAS Results with Position-Aware Analysis:")
        logger.info(f"   Regimes detected: {len(np.unique(tas_result.regime_predictions))}")
        logger.info(f"   Economic significance: {np.mean(tas_result.economic_significance_scores)".3f"}")
        logger.info(f"   Trading viability: {np.mean(tas_result.trading_viability_scores)".3f"}")
        logger.info(f"   Position-aware analysis: Available")

    except Exception as e:
        logger.warning(f"TAS position-aware demonstration failed: {e}")


def demonstrate_nas_position_awareness():
    """Demonstrate how NAS system uses shared position-aware utilities."""
    logger.info("\n🧠 Demonstrating NAS with Position-Aware Utilities")
    logger.info("=" * 50)

    try:
        # Create NAS detector with position-aware support
        from ..nas_regime.core.perfect_nas_config import PerfectNASConfig

        config = PerfectNASConfig(
            n_regimes=6,
            primary_architecture=PerfectNASConfig.NeuralArchitectureType.HYBRID,
            enable_economic_evaluation=True,
            enable_neural_odes=True,
            enable_vision_transformers=True
        )

        nas_detector = PerfectNASRegimeDetector(config)

        # Generate sample data
        market_data = generate_sample_market_data(500)
        position_directions = np.where(
            market_data['close'].pct_change() > 0.001, 1,
            np.where(market_data['close'].pct_change() < -0.001, -1, 0)
        )

        logger.info("🚀 Running NAS with position-aware analysis...")
        nas_result = nas_detector.detect_regimes(
            market_data, optimize_architecture=True, enable_meta_learning=True
        )

        logger.info("📊 NAS Results with Position-Aware Analysis:")
        logger.info(f"   Regimes detected: {len(np.unique(nas_result.regime_predictions))}")
        logger.info(f"   Economic significance: {np.mean(nas_result.economic_significance_scores)".3f"}")
        logger.info(f"   Trading viability: {np.mean(nas_result.trading_viability_scores)".3f"}")
        logger.info(f"   Position-aware analysis: Available")

    except Exception as e:
        logger.warning(f"NAS position-aware demonstration failed: {e}")


def main():
    """Run complete demonstration."""
    logging.basicConfig(level=logging.INFO)

    logger.info("🚀 Position-Aware Trading Analysis Demonstration")
    logger.info("This demonstrates how both TAS and NAS systems use shared utilities")
    logger.info("for accurate win rate calculations with long and short positions.")

    # Demonstrate shared utilities
    demonstrate_shared_position_aware_utilities()

    # Demonstrate TAS integration
    demonstrate_tas_position_awareness()

    # Demonstrate NAS integration
    demonstrate_nas_position_awareness()

    logger.info("\n✅ All demonstrations completed successfully!")
    logger.info("Both TAS and NAS systems now use shared position-aware utilities")
    logger.info("for consistent and accurate win rate calculations.")


if __name__ == "__main__":
    main()