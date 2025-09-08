#!/usr/bin/env python3
"""
Demo script for Step08 Enhanced Regime Data Splitting Reporting

This script demonstrates the enhanced reporting capabilities for Step08,
which handles regime data splitting and unified dataset creation with HMM composite clusters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.steps.step08_enhanced_reporting import Step08EnhancedReporter
from src.utils.logger import system_logger
import logging
import time

def create_sample_market_data(num_samples: int = 10000) -> pd.DataFrame:
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

    # Add some technical indicators
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()

    # Add composite cluster IDs (HMM regime labels)
    # Simulate different market regimes
    regime_patterns = []
    current_regime = 0

    for i in range(len(data)):
        # Change regime occasionally
        if np.random.random() < 0.001:  # 0.1% chance to change regime
            current_regime = (current_regime + 1) % 5  # 5 different regimes

        regime_patterns.append(current_regime)

    data['composite_cluster_id'] = regime_patterns

    # Set timestamp as index
    data = data.set_index('timestamp')

    return data

def create_sample_regime_clusters(data: pd.DataFrame) -> list:
    """Create sample unique regime cluster IDs."""
    return sorted(data['composite_cluster_id'].unique())

def demonstrate_enhanced_reporting():
    """Demonstrate the Step08 enhanced reporting system."""
    logger = system_logger.getChild('Step08.Demo')
    logger.info("🚀 Starting Step08 Enhanced Reporting Demonstration")

    try:
        # Configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache',
            'lookback_days': 365
        }

        # Create sample data
        logger.info("📊 Creating sample market data...")
        sample_data = create_sample_market_data(5000)
        unique_clusters = create_sample_regime_clusters(sample_data)

        logger.info(f"✅ Created sample data: {len(sample_data)} rows, {len(unique_clusters)} regimes")
        logger.info(f"   Date range: {sample_data.index.min()} to {sample_data.index.max()}")
        logger.info(f"   Regimes: {unique_clusters}")

        # Initialize enhanced reporter
        logger.info("🔧 Initializing Step08 Enhanced Reporter...")
        reporter = Step08EnhancedReporter(config)

        # Prepare execution metadata
        execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': 45.67,
            'memory_usage_mb': 234.56,
            'cpu_usage_percent': 67.89,
            'data_quality_score': 0.945,
            'processing_efficiency': 0.892,
            'total_samples': len(sample_data)
        }

        # Prepare validation results
        validation_results = {
            'validation_passed': True,
            'data_loaded': True,
            'regime_column_present': True,
            'sufficient_data': True,
            'temporal_ordering': True,
            'errors': [],
            'warnings': ['Minor data quality issue detected in 0.02% of samples'],
            'schema_validation': {
                'required_columns_present': True,
                'data_types_correct': True,
                'index_valid': True
            },
            'temporal_validation': {
                'no_future_dates': True,
                'reasonable_time_range': True,
                'consistent_intervals': True
            },
            'integrity_checks': {
                'no_duplicate_timestamps': True,
                'data_integrity': True,
                'regime_consistency': True
            }
        }

        # Generate comprehensive report
        logger.info("📈 Generating comprehensive Step08 analysis report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            unified_data=sample_data,
            unique_clusters=unique_clusters,
            execution_metadata=execution_metadata,
            validation_results=validation_results
        )

        # Display key metrics
        logger.info("📊 Key Analysis Results:")
        logger.info(f"   📈 Total Regimes: {comprehensive_report['regime_distribution_analysis']['total_regimes']}")
        logger.info(f"   ⚖️ Data Balance Score: {comprehensive_report['regime_distribution_analysis']['data_balance_score']:.3f}")
        logger.info(f"   🎯 Overall Quality Score: {comprehensive_report['data_quality_analysis']['overall_quality_score']:.3f}")
        logger.info(f"   ⚡ Execution Time: {comprehensive_report['performance_analysis']['execution_time_seconds']:.2f}s")
        logger.info(f"   💾 Memory Usage: {comprehensive_report['performance_analysis']['memory_usage_mb']:.2f}MB")

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
        logger.info("🎉 Step08 Enhanced Reporting Demo Summary")
        logger.info("="*60)
        logger.info("✅ Successfully demonstrated enhanced regime data splitting analysis")
        logger.info("✅ Generated comprehensive reports with multiple formats:")
        logger.info("   • JSON: Detailed structured data")
        logger.info("   • Markdown: Human-readable summary")
        logger.info("   • CSV: Key metrics for analysis")
        logger.info("   • PNG: Visual charts and graphs")
        logger.info("✅ Analyzed regime distribution, data quality, and performance")
        logger.info("✅ Provided actionable recommendations and alerts")
        logger.info("✅ Demonstrated fallback mechanisms for robustness")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger = system_logger.getChild('Step08.Demo.Main')
    logger.info("🎯 Starting Step08 Enhanced Reporting Demonstration")
    logger.info("="*60)

    success = demonstrate_enhanced_reporting()

    if success:
        logger.info("🎉 Demonstration completed successfully!")
        logger.info("📚 Check the generated report files in src/training/reports/step08/")
    else:
        logger.error("❌ Demonstration failed - check logs for details")
        sys.exit(1)
