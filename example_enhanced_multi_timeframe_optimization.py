"""
Example: Enhanced Multi-Timeframe Optimization with Optimized Lookback Periods

This script demonstrates how to use the enhanced multi-timeframe optimizer
that integrates with the matrix optimization system to use optimized lookback
periods instead of fixed periods for multi-timeframe and cross-timeframe features.
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Import the enhanced multi-timeframe optimizer
from src.training.enhanced_multi_timeframe_optimizer import (
    EnhancedMultiTimeframeOptimizer,
    OptimizedTimeframeConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    logger.info(f"📊 Creating sample data with {n_samples} samples...")

    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(n_samples)]

    # Generate realistic price data with trends and volatility
    np.random.seed(42)

    # Base price with trend
    base_price = 100 + np.cumsum(np.random.normal(0, 0.1, n_samples))

    # Add volatility clusters
    volatility = np.random.gamma(2, 0.5, n_samples)
    returns = np.random.normal(0, 0.01, n_samples) * volatility

    # Calculate OHLCV
    close_prices = base_price * np.exp(np.cumsum(returns))
    open_prices = close_prices * np.exp(np.random.normal(0, 0.005, n_samples))
    high_prices = np.maximum(open_prices, close_prices) * np.exp(np.abs(np.random.normal(0, 0.01, n_samples)))
    low_prices = np.minimum(open_prices, close_prices) * np.exp(-np.abs(np.random.normal(0, 0.01, n_samples)))

    # Generate volume with some correlation to price movement
    volume = np.random.lognormal(10, 0.5, n_samples) * (1 + np.abs(returns) * 10)

    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=timestamps)

    logger.info(f"✅ Created sample data with shape: {data.shape}")
    return data

def create_sample_matrix_optimization_results() -> dict:
    """Create sample matrix optimization results for demonstration."""
    logger.info("🔧 Creating sample matrix optimization results...")

    # Simulate matrix optimization results with diverse lookback periods
    matrix_results = {
        "diverse_lookback_periods": {
            "RSI": {
                "selected_periods": [7, 14, 21],
                "diversity_score": 0.65,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "MACD_fast": {
                "selected_periods": [8, 12, 16],
                "diversity_score": 0.58,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "Bollinger_Bands": {
                "selected_periods": [10, 20, 30],
                "diversity_score": 0.72,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "SMA": {
                "selected_periods": [5, 20, 50],
                "diversity_score": 0.81,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "EMA": {
                "selected_periods": [5, 20, 50],
                "diversity_score": 0.79,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "ATR": {
                "selected_periods": [10, 20, 30],
                "diversity_score": 0.68,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "VWAP": {
                "selected_periods": [5, 10, 20],
                "diversity_score": 0.55,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "VWAP_Momentum": {
                "selected_periods": [5, 10, 20],
                "diversity_score": 0.62,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            },
            "VWAP_Volatility": {
                "selected_periods": [5, 10, 20],
                "diversity_score": 0.59,
                "optimization_method": "matrix",
                "quality_check": "passed_3_periods"
            }
        },
        "regime_specific_periods": {
            "regime_0": {  # Low volatility regime
                "RSI": {
                    "selected_periods": [5, 10, 15],
                    "diversity_score": 0.61
                },
                "VWAP": {
                    "selected_periods": [4, 8, 15],
                    "diversity_score": 0.58
                }
            },
            "regime_1": {  # Medium volatility regime
                "RSI": {
                    "selected_periods": [7, 14, 21],
                    "diversity_score": 0.65
                },
                "VWAP": {
                    "selected_periods": [5, 10, 20],
                    "diversity_score": 0.62
                }
            },
            "regime_2": {  # High volatility regime
                "RSI": {
                    "selected_periods": [10, 20, 30],
                    "diversity_score": 0.69
                },
                "VWAP": {
                    "selected_periods": [8, 15, 25],
                    "diversity_score": 0.66
                }
            }
        }
    }

    logger.info(f"✅ Created sample matrix results with {len(matrix_results['diverse_lookback_periods'])} optimized features")
    return matrix_results

def create_sample_regime_labels(n_samples: int) -> pd.Series:
    """Create sample regime labels for demonstration."""
    logger.info("🏛️ Creating sample regime labels...")

    # Create regime labels with some persistence
    np.random.seed(42)
    regimes = []
    current_regime = 0

    for i in range(n_samples):
        # Regime persistence with occasional switches
        if np.random.random() < 0.98:  # 98% chance to stay in same regime
            regimes.append(current_regime)
        else:
            current_regime = np.random.randint(0, 3)  # 3 regimes: 0, 1, 2
            regimes.append(current_regime)

    regime_series = pd.Series(regimes, name='regime')
    logger.info(f"✅ Created regime labels: {regime_series.value_counts().to_dict()}")
    return regime_series

async def demonstrate_enhanced_multi_timeframe_optimization():
    """Demonstrate the enhanced multi-timeframe optimization process."""
    logger.info("🚀 Starting Enhanced Multi-Timeframe Optimization Demonstration")

    # 1. Create sample data
    data = create_sample_data(5000)

    # 2. Create sample target variable (future returns)
    target = data['close'].pct_change(5).shift(-5).fillna(0)

    # 3. Create sample matrix optimization results
    matrix_results = create_sample_matrix_optimization_results()

    # 4. Create sample regime labels
    regime_labels = create_sample_regime_labels(len(data))

    # 5. Configure the enhanced multi-timeframe optimizer
    config = OptimizedTimeframeConfig(
        base_timeframes=["1m", "5m", "15m", "30m", "1h"],
        cross_timeframe_enabled=True,
        regime_specific=True,
        quality_thresholds={
            "min_correlation": 0.1,  # Lower threshold for demo
            "max_correlation": 0.8,
            "min_information_score": 0.01,  # Lower threshold for demo
            "min_diversity_score": 0.1  # Lower threshold for demo
        }
    )

    # 6. Initialize the enhanced optimizer
    optimizer = EnhancedMultiTimeframeOptimizer(config, matrix_results)

    # 7. Generate optimized multi-timeframe features
    logger.info("🔧 Generating optimized multi-timeframe features...")
    optimized_features = await optimizer.generate_optimized_multi_timeframe_features(
        data=data,
        target=target,
        regime_labels=regime_labels
    )

    # 8. Analyze results
    logger.info("📊 Analyzing optimization results...")

    # Count features by type
    feature_types = {}
    for feature_name in optimized_features.keys():
        if '_' in feature_name:
            parts = feature_name.split('_')
            if len(parts) >= 3:
                indicator = parts[0]
                timeframe = parts[-1] if parts[-1] in ['1m', '5m', '15m', '30m', '1h'] else 'cross'

                if indicator not in feature_types:
                    feature_types[indicator] = {'timeframes': {}, 'cross': 0}

                if timeframe == 'cross':
                    feature_types[indicator]['cross'] += 1
                else:
                    if timeframe not in feature_types[indicator]['timeframes']:
                        feature_types[indicator]['timeframes'][timeframe] = 0
                    feature_types[indicator]['timeframes'][timeframe] += 1

    # 9. Print results
    logger.info("📈 ENHANCED MULTI-TIMEFRAME OPTIMIZATION RESULTS")
    logger.info("=" * 60)

    logger.info(f"Total optimized features generated: {len(optimized_features)}")
    logger.info(f"Matrix optimization features used: {len(matrix_results['diverse_lookback_periods'])}")
    logger.info(f"Regime-specific features: {len([f for f in optimized_features.keys() if f.startswith('regime_')])}")
    logger.info(f"Cross-timeframe features: {len([f for f in optimized_features.keys() if 'diff_' in f or 'ratio_' in f])}")

    logger.info("\n📊 Feature Distribution by Indicator:")
    for indicator, info in feature_types.items():
        logger.info(f"  {indicator}:")
        for timeframe, count in info['timeframes'].items():
            logger.info(f"    {timeframe}: {count} features")
        if info['cross'] > 0:
            logger.info(f"    Cross-timeframe: {info['cross']} features")

    # 10. Show sample features
    logger.info("\n🔍 Sample Optimized Features:")
    sample_features = list(optimized_features.keys())[:10]
    for feature in sample_features:
        logger.info(f"  - {feature}")

    # 11. Save results
    output_path = Path("data/optimization_results")
    output_path.mkdir(parents=True, exist_ok=True)
    optimizer.save_optimization_results(str(output_path))

    # 12. Compare with traditional approach
    logger.info("\n🔄 COMPARISON WITH TRADITIONAL APPROACH")
    logger.info("=" * 60)

    # Traditional approach would use fixed periods
    traditional_periods = [5, 10, 20]  # Fixed periods
    traditional_features = len(traditional_periods) * len(config.base_timeframes) * len(matrix_results['diverse_lookback_periods'])

    logger.info(f"Traditional approach (fixed periods): ~{traditional_features} features")
    logger.info(f"Enhanced approach (optimized periods): {len(optimized_features)} features")
    logger.info(f"Improvement: {len(optimized_features) / traditional_features:.1f}x more features")

    # 13. Quality metrics
    logger.info("\n🎯 QUALITY METRICS")
    logger.info("=" * 60)

    # Calculate average correlation with target
    correlations = []
    for feature_name, feature_series in optimized_features.items():
        if isinstance(feature_series, pd.Series):
            corr = abs(feature_series.corr(target))
            if not pd.isna(corr):
                correlations.append(corr)

    if correlations:
        avg_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)
        min_correlation = np.min(correlations)

        logger.info(f"Average correlation with target: {avg_correlation:.3f}")
        logger.info(f"Maximum correlation with target: {max_correlation:.3f}")
        logger.info(f"Minimum correlation with target: {min_correlation:.3f}")

    # 14. Key benefits summary
    logger.info("\n✨ KEY BENEFITS OF ENHANCED MULTI-TIMEFRAME OPTIMIZATION")
    logger.info("=" * 60)

    benefits = [
        "✅ Uses optimized lookback periods from matrix optimization instead of fixed periods",
        "✅ Generates diverse cross-timeframe features with optimized period pairs",
        "✅ Includes regime-specific features with regime-optimized periods",
        "✅ Quality filtering ensures only high-value features are retained",
        "✅ Backward compatible with existing multi-timeframe systems",
        "✅ Integrates seamlessly with matrix optimization results",
        "✅ Supports multiple timeframes (1m, 5m, 15m, 30m, 1h)",
        "✅ Generates momentum, volatility, volume, and range features",
        "✅ Automatic feature alignment and quality validation"
    ]

    for benefit in benefits:
        logger.info(benefit)

    logger.info("\n🎉 Enhanced Multi-Timeframe Optimization Demonstration Complete!")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_enhanced_multi_timeframe_optimization())