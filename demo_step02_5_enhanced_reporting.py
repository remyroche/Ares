#!/usr/bin/env python3
"""
Demonstration of Step 2.5 Enhanced Reporting System

This script demonstrates the comprehensive reporting capabilities
for step02_5_sr_optimization with detailed metrics and visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.training.steps.data_collection.data_preparation.step02_5_enhanced_reporting import Step02_5EnhancedReporter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    # Create sample OHLCV data
    np.random.seed(42)
    n_rows = 1000

    # Generate timestamps
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(minutes=i*5) for i in range(n_rows)]

    # Generate realistic price data with trends
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(n_rows):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 50)  # Slow trend
        noise = np.random.normal(0, 0.005)  # Random noise
        current_price *= (1 + trend + noise)
        prices.append(current_price)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(n_rows)],
        'vwap': prices,  # Simplified VWAP
        'rsi': [np.random.uniform(20, 80) for _ in range(n_rows)],
        'macd': [np.random.normal(0, 10) for _ in range(n_rows)],
        'bb_upper': [p * 1.02 for p in prices],
        'bb_lower': [p * 0.98 for p in prices],
        'stoch_k': [np.random.uniform(0, 100) for _ in range(n_rows)],
        'stoch_d': [np.random.uniform(0, 100) for _ in range(n_rows)]
    })

    # Add some NaN values to demonstrate data quality assessment
    nan_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[nan_indices, 'volume'] = np.nan

    return df


def create_sample_sr_levels():
    """Create sample S/R levels for demonstration."""
    return {
        'support_levels': [
            {'price': 48500, 'strength': 0.85, 'touches': 12, 'bounces': 8, 'bounce_rate': 0.67},
            {'price': 47200, 'strength': 0.72, 'touches': 9, 'bounces': 5, 'bounce_rate': 0.56},
            {'price': 45800, 'strength': 0.68, 'touches': 7, 'bounces': 4, 'bounce_rate': 0.57},
            {'price': 44500, 'strength': 0.91, 'touches': 15, 'bounces': 12, 'bounce_rate': 0.80},
            {'price': 43200, 'strength': 0.76, 'touches': 11, 'bounces': 7, 'bounce_rate': 0.64}
        ],
        'resistance_levels': [
            {'price': 51500, 'strength': 0.88, 'touches': 14, 'bounces': 10, 'bounce_rate': 0.71},
            {'price': 52800, 'strength': 0.79, 'touches': 10, 'bounces': 6, 'bounce_rate': 0.60},
            {'price': 54100, 'strength': 0.83, 'touches': 13, 'bounces': 9, 'bounce_rate': 0.69},
            {'price': 55400, 'strength': 0.95, 'touches': 18, 'bounces': 15, 'bounce_rate': 0.83},
            {'price': 56700, 'strength': 0.74, 'touches': 8, 'bounces': 5, 'bounce_rate': 0.63}
        ]
    }


def create_sample_ml_results():
    """Create sample ML results for demonstration."""
    return {
        'model_type': 'RandomForestClassifier',
        'direction_accuracy': 0.847,
        'volatility_mae': 0.0234,
        'training_samples': 800,
        'test_samples': 200,
        'feature_count': 45,
        'training_time': 2.34,
        'cross_validation_scores': [0.82, 0.85, 0.83, 0.86, 0.84],
        'feature_importance': {
            'rsi': 0.142,
            'macd': 0.118,
            'bb_position': 0.098,
            'volume_price_trend': 0.087,
            'stoch_k': 0.076,
            'distance_to_support': 0.069,
            'vwap_deviation': 0.065,
            'price_momentum_20': 0.058,
            'cci': 0.052,
            'williams_r': 0.048,
            'price_acceleration_10': 0.043,
            'volatility_20': 0.038,
            'on_balance_volume': 0.032,
            'mfi': 0.029,
            'roc_20': 0.025
        },
        'classification_report': {
            'weighted avg': {
                'precision': 0.851,
                'recall': 0.847,
                'f1-score': 0.846,
                'support': 200
            }
        },
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }


def create_sample_execution_data():
    """Create sample execution data for demonstration."""
    return {
        'execution_time': 124.67,
        'memory_usage': 2100.5,
        'cpu_usage': 87.3,
        'function_calls': 15420,
        'step_breakdown': {
            'data_loading': 12.34,
            'feature_engineering': 45.67,
            'sr_detection': 23.89,
            'ml_training': 34.56,
            'report_generation': 8.21
        },
        'performance_summary': {
            'features_per_second': 358.9,
            'sr_levels_per_second': 2.4,
            'ml_accuracy': 0.847
        },
        'feature_count': 45,
        'data_rows': 1000,
        'sr_levels_detected': 10,
        'ml_accuracy': 0.847,
        'processing_timestamp': datetime.now().isoformat()
    }


def main():
    """Main demonstration function."""
    logger.info("🚀 Step 2.5 Enhanced Reporting System Demonstration")
    logger.info("=" * 60)

    try:
        # Create sample data
        logger.info("📊 Creating sample market data...")
        sample_data = create_sample_data()
        logger.info(f"✅ Created {len(sample_data)} rows of sample data")

        # Create sample S/R levels
        logger.info("🎯 Creating sample S/R levels...")
        sr_levels = create_sample_sr_levels()
        logger.info(f"✅ Created {len(sr_levels['support_levels'])} support and {len(sr_levels['resistance_levels'])} resistance levels")

        # Create sample ML results
        logger.info("🤖 Creating sample ML results...")
        ml_results = create_sample_ml_results()
        logger.info(f"✅ ML Direction Accuracy: {ml_results['direction_accuracy']:.3f}")
        # Create sample execution data
        logger.info("⚡ Creating sample execution data...")
        execution_data = create_sample_execution_data()
        logger.info(f"✅ Execution Time: {execution_data['execution_time']:.2f}s")
        # Initialize enhanced reporter
        logger.info("📝 Initializing enhanced reporter...")
        reporter = Step02_5EnhancedReporter(
            symbol="BTCUSDT",
            exchange="BINANCE",
            timeframe="5m"
        )

        # Generate comprehensive report
        logger.info("🎯 Generating comprehensive report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            sr_levels=sr_levels,
            ml_results=ml_results,
            execution_data=execution_data,
            data=sample_data
        )

        logger.info("💾 Saving comprehensive reports...")
        saved_reports = reporter.save_comprehensive_report(
            report_data=comprehensive_report,
            include_visualizations=True
        )

        # Display results
        logger.info("\n🎉 Enhanced Reporting Demonstration Complete!")
        logger.info("=" * 60)
        logger.info("📁 Generated Reports:")
        for report_type, file_path in saved_reports.items():
            logger.info(f"  • {report_type.replace('_', ' ').title()}: {file_path}")

        # Show key metrics from the report
        metadata = comprehensive_report.get('report_metadata', {})
        perf = comprehensive_report.get('performance_metrics', {})
        dq = comprehensive_report.get('data_quality_assessment', {})
        sr = comprehensive_report.get('sr_level_analysis', {})
        ml = comprehensive_report.get('ml_model_insights', {})

        logger.info("\n📊 Key Metrics Summary:")
        logger.info(f"  • Symbol: {metadata.get('symbol')}")
        logger.info(f"✅ Execution Time: {execution_data['execution_time']:.2f}s")
        logger.info(f"✅ Memory Usage: {execution_data['memory_usage']:.1f} MB")
        logger.info(f"  • Data Completeness: {dq.get('data_completeness_score', 0):.1f}%")
        support_levels = sr.get('support_analysis', {}).get('total_levels', 0)
        resistance_levels = sr.get('resistance_analysis', {}).get('total_levels', 0)
        logger.info(f"  • S/R Levels Detected: {support_levels + resistance_levels}")
        logger.info(f"✅ ML Direction Accuracy: {ml_results['direction_accuracy']:.3f}")
        logger.info(f"  • ML Features Used: {ml.get('feature_count', 0)}")
        logger.info(f"  • Risk Assessment: {comprehensive_report.get('risk_assessment', 'UNKNOWN')}")

        logger.info("\n✅ Demonstration completed successfully!")
        logger.info("📖 Check the generated reports for detailed analysis and visualizations.")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
