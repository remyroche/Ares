#!/usr/bin/env python3
"""
Demo script for Step06 Enhanced Advanced Feature Engineering Reporting

This script demonstrates the enhanced reporting capabilities for Step06,
which handles advanced feature engineering with wavelet transforms, multi-timeframe analysis,
hardware acceleration, and comprehensive technical indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.feature_engineering.step06_enhanced_reporting import Step06EnhancedReporter
from src.utils.logger import system_logger

def create_sample_market_data(num_samples: int = 5000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)

    # Create datetime index
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=i) for i in range(num_samples)]

    # Generate realistic OHLCV data
    base_price = 50000  # Base price for crypto-like data

    # Create trending and volatile periods
    trend_changes = np.random.choice([-1, 0, 1], size=num_samples, p=[0.3, 0.4, 0.3])
    volatility_changes = np.random.choice([0.5, 1.0, 2.0], size=num_samples, p=[0.5, 0.3, 0.2])

    prices = [base_price]
    volumes = []

    for i in range(1, num_samples):
        # Price movement based on trend and volatility
        price_change = np.random.normal(0, 0.001 * volatility_changes[i])
        price_change += trend_changes[i] * 0.0005  # Add trend influence

        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 0.1))  # Prevent negative prices

        # Volume based on price volatility
        volume = np.random.normal(1000, 200) * (1 + abs(price_change) * 100)
        volumes.append(max(volume, 10))

    # Create OHLC from price series
    opens = prices[:-1]
    highs = [max(o, c) + np.random.uniform(0, o * 0.002) for o, c in zip(opens, prices[1:])]
    lows = [min(o, c) - np.random.uniform(0, o * 0.002) for o, c in zip(opens, prices[1:])]
    closes = prices[1:]
    volumes = volumes[:len(closes)]  # Ensure same length

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates[:-1],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })

    # Add some basic technical indicators to simulate labeled data from step05
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = 50 + np.random.normal(0, 10, len(data))  # Mock RSI
    data['macd'] = data['close'] - data['sma_20']  # Mock MACD
    data['bb_upper'] = data['sma_20'] + 2 * data['volatility']  # Mock Bollinger Bands
    data['bb_lower'] = data['sma_20'] - 2 * data['volatility']

    # Add target labels (simulating step05 output)
    data['target'] = np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3])
    data['sample_weight'] = np.random.uniform(0.5, 1.5, len(data))

    # Set timestamp as index
    data = data.set_index('timestamp')

    return data

def create_sample_engineered_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample engineered features to simulate Step06 output."""
    # Start with input data
    features = input_data.copy()

    # Add wavelet features (simulated)
    for level in range(3):
        for component in ['approx', 'detail']:
            features[f'wavelet_l{level}_{component}_close'] = features['close'] + np.random.normal(0, 0.01, len(features))
            features[f'wavelet_l{level}_{component}_volume'] = features['volume'] + np.random.normal(0, 10, len(features))

    # Add multi-timeframe features (simulated)
    timeframes = ['5m', '15m', '1h', '4h']
    for tf in timeframes:
        features[f'mtf_{tf}_sma'] = features['close'].rolling(5).mean() + np.random.normal(0, 5, len(features))
        features[f'mtf_{tf}_volatility'] = features['volatility'].rolling(5).std() + np.random.normal(0, 0.01, len(features))

    # Add microstructure features
    features['spread'] = (features['high'] - features['low']) / features['close']
    features['realized_volatility'] = features['returns'].rolling(10).std()
    features['price_impact'] = features['volume'] * features['returns']

    # Add correlation features
    features['price_volume_corr'] = features['close'].rolling(20).corr(features['volume'])
    features['high_low_corr'] = features['high'].rolling(20).corr(features['low'])

    # Add momentum features
    features['momentum_5'] = features['close'] - features['close'].shift(5)
    features['momentum_10'] = features['close'] - features['close'].shift(10)
    features['momentum_20'] = features['close'] - features['close'].shift(20)

    # Add regime-aware features (simulated)
    features['regime_volatility'] = features['volatility'] * np.random.uniform(0.8, 1.2, len(features))
    features['regime_trend'] = features['sma_20'] * np.random.uniform(0.9, 1.1, len(features))

    # Add interaction features
    features['vol_price_interaction'] = features['volatility'] * features['close']
    features['momentum_vol_interaction'] = features['momentum_5'] * features['volatility']

    # Remove target and sample_weight as they're not features
    features = features.drop(['target', 'sample_weight'], axis=1, errors='ignore')

    return features

def demonstrate_enhanced_reporting():
    """Demonstrate the Step06 enhanced reporting system."""
    logger = system_logger.getChild('Step06.Demo')
    logger.info("🚀 Starting Step06 Enhanced Reporting Demonstration")

    try:
        # Configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache',
            'feature_engineering': {
                'enable_wavelets': True,
                'enable_multi_timeframe': True,
                'enable_feature_interactions': True,
                'timeframes': ['5m', '15m', '1h', '4h'],
                'max_features': 500,
                'wavelet_config': {
                    'levels': 3,
                    'family': 'db4',
                    'decomposition_levels': 4
                }
            }
        }

        # Create sample data
        logger.info("📊 Creating sample market data...")
        input_data = create_sample_market_data(5000)
        output_features = create_sample_engineered_features(input_data)

        logger.info(f"✅ Created sample data: {len(input_data)} rows")
        logger.info(f"   Input features: {input_data.shape[1]}")
        logger.info(f"   Output features: {output_features.shape[1]}")
        logger.info(f"   Features created: {output_features.shape[1] - input_data.shape[1]}")

        # Initialize enhanced reporter
        logger.info("🔧 Initializing Step06 Enhanced Reporter...")
        reporter = Step06EnhancedReporter(config)

        # Prepare feature configuration
        feature_config = config['feature_engineering']

        # Prepare execution metadata
        execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_execution_time': 125.67,
            'features_created': output_features.shape[1] - input_data.shape[1],
            'chunk_processing_metrics': {
                'chunks_processed': 12,
                'avg_chunk_time': 10.47,
                'max_memory_usage': 2048
            },
            'caching_efficiency': 0.92
        }

        # Prepare hardware metrics
        hardware_metrics = {
            'gpu_utilization': 0.87,
            'cpu_utilization': 0.78,
            'vectorization_efficiency': 0.94,
            'memory_usage_mb': 2156.0,
            'processing_speedup': 2.8,
            'optimization_enabled': True,
            'm1_gpu_available': True,
            'vectorized_operations': 15000,
            'parallel_processing_efficiency': 0.91
        }

        # Generate comprehensive report
        logger.info("📈 Generating comprehensive Step06 analysis report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            input_data=input_data,
            output_features=output_features,
            feature_config=feature_config,
            execution_metadata=execution_metadata,
            hardware_metrics=hardware_metrics
        )

        # Display key metrics
        logger.info("📊 Key Analysis Results:")
        logger.info(f"   🔧 Total Features Created: {comprehensive_report['feature_engineering_analysis']['total_features_created']}")
        logger.info(f"   🌊 Wavelet Features: {comprehensive_report['feature_engineering_analysis']['wavelet_features_count']}")
        logger.info(f"   ⏰ Multi-Timeframe Features: {comprehensive_report['feature_engineering_analysis']['multi_timeframe_features_count']}")
        logger.info(f"   📈 Technical Indicators: {comprehensive_report['feature_engineering_analysis']['technical_indicators_count']}")
        logger.info(f"   ⚡ Hardware Acceleration Score: {comprehensive_report['hardware_acceleration_analysis']['hardware_acceleration_score']:.3f}")
        logger.info(f"   🎯 Feature Quality Score: {comprehensive_report['feature_quality_analysis']['overall_quality_score']:.3f}")
        logger.info(f"   🚀 Processing Speedup: {comprehensive_report['hardware_acceleration_analysis']['processing_speedup']:.1f}x")

        # Display recommendations and alerts
        if 'recommendations' in comprehensive_report:
            logger.info("💡 Recommendations:")
            for rec in comprehensive_report['recommendations'][:3]:  # Show first 3
                logger.info(f"   • {rec}")

        if 'alerts' in comprehensive_report:
            logger.info("🚨 Alerts:")
            for alert in comprehensive_report['alerts'][:3]:  # Show first 3
                logger.info(f"   • {alert}")

        # Save comprehensive reports
        logger.info("💾 Saving comprehensive reports...")
        saved_files = reporter.save_comprehensive_report(
            report_data=comprehensive_report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        logger.info("✅ Demo completed successfully!")
        logger.info(f"📄 Generated {len(saved_files)} report files:")
        for file_path in saved_files:
            logger.info(f"   • {file_path}")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 Step06 Enhanced Reporting Demo Summary")
        logger.info("="*60)
        logger.info("✅ Successfully demonstrated enhanced feature engineering analysis")
        logger.info("✅ Generated comprehensive reports with multiple formats:")
        logger.info("   • JSON: Detailed structured data")
        logger.info("   • Markdown: Human-readable summary")
        logger.info("   • CSV: Key metrics for analysis")
        logger.info("   • PNG: Visual charts and graphs")
        logger.info("✅ Analyzed feature engineering, hardware acceleration, and quality")
        logger.info("✅ Provided actionable recommendations and alerts")
        logger.info("✅ Demonstrated wavelet, multi-timeframe, and interaction analysis")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger = system_logger.getChild('Step06.Demo.Main')
    logger.info("🎯 Starting Step06 Enhanced Reporting Demonstration")
    logger.info("="*60)

    success = demonstrate_enhanced_reporting()

    if success:
        logger.info("🎉 Demonstration completed successfully!")
        logger.info("📚 Check the generated report files in src/training/reports/step06/")
    else:
        logger.error("❌ Demonstration failed - check logs for details")
        sys.exit(1)
