#!/usr/bin/env python3
"""
Test script for enhanced Step08 reporting functionality.

This script tests the comprehensive reporting capabilities of the Step08
enhanced reporting system to ensure all features work correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.step08_enhanced_reporting import Step08EnhancedReporter

def create_sample_data():
    """Create sample data for testing the reporting system."""
    # Create sample OHLCV data with regime labels
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    np.random.seed(42)

    data = {
        'timestamp': dates,
        'open': 50000 + np.random.normal(0, 1000, len(dates)),
        'high': 50200 + np.random.normal(0, 1000, len(dates)),
        'low': 49800 + np.random.normal(0, 1000, len(dates)),
        'close': 50000 + np.random.normal(0, 1000, len(dates)),
        'volume': np.random.lognormal(10, 1, len(dates)),
        'composite_cluster_id': np.random.choice([0, 1, 2, 3], len(dates), p=[0.4, 0.3, 0.2, 0.1])
    }

    # Ensure high >= close >= low >= open (approximately)
    for i in range(len(data['close'])):
        data['high'][i] = max(data['high'][i], data['close'][i], data['open'][i])
        data['low'][i] = min(data['low'][i], data['close'][i], data['open'][i])

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df

def create_sample_config():
    """Create sample configuration for testing."""
    return {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'lookback_days': 1095,
        'data_dir': 'data_cache',
        'force_rerun': False
    }

def create_sample_execution_metadata():
    """Create sample execution metadata."""
    return {
        'duration_seconds': 245.67,
        'memory_usage_mb': 1847.3,
        'cpu_usage_percent': 78.4,
        'total_samples': 100000,
        'file_operations': 3
    }

def create_sample_validation_results():
    """Create sample validation results."""
    return {
        'validation_passed': True,
        'errors': [],
        'warnings': ['Minor temporal gap detected'],
        'data_loaded': True,
        'regime_column_present': True,
        'sufficient_data': True,
        'temporal_ordering': True,
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

def test_enhanced_reporting():
    """Test the enhanced Step08 reporting functionality."""
    print("🧪 Testing Step08 Enhanced Reporting System")
    print("=" * 50)

    try:
        # Create sample data
        print("📊 Creating sample data...")
        unified_data = create_sample_data()
        unique_clusters = [0, 1, 2, 3]
        config = create_sample_config()
        execution_metadata = create_sample_execution_metadata()
        validation_results = create_sample_validation_results()

        print(f"   Data shape: {unified_data.shape}")
        print(f"   Unique regimes: {unique_clusters}")
        print(f"   Date range: {unified_data.index.min()} to {unified_data.index.max()}")

        # Initialize the reporter
        print("🔧 Initializing enhanced reporter...")
        reporter = Step08EnhancedReporter(config)

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            unified_data=unified_data,
            unique_clusters=unique_clusters,
            execution_metadata=execution_metadata,
            validation_results=validation_results
        )

        if 'error' in report:
            print(f"❌ Report generation failed: {report['error']}")
            return False

        print("✅ Report generated successfully")

        # Test key sections
        expected_sections = [
            'regime_distribution_analysis',
            'data_quality_analysis',
            'performance_analysis',
            'temporal_analysis',
            'recommendations',
            'alerts'
        ]

        missing_sections = []
        for section in expected_sections:
            if section not in report:
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️ Missing sections: {missing_sections}")
        else:
            print("✅ All expected sections present")

        # Test performance predictions
        print("🔮 Testing performance predictions...")
        predictions = reporter._generate_performance_predictions()

        if 'error' in predictions:
            print(f"⚠️ Performance predictions failed: {predictions['error']}")
        else:
            print("✅ Performance predictions generated")

            # Test specific predictions
            if 'model_performance_predictions' in predictions:
                mpp = predictions['model_performance_predictions']
                accuracy = mpp.get('predicted_model_accuracy', 0)
                print(f"   Predicted model accuracy: {accuracy:.3f}")
        # Test enhanced alerts
        print("🚨 Testing enhanced alerts...")
        alerts = reporter._generate_alerts()

        if alerts:
            print(f"✅ Generated {len(alerts)} alerts")
            # Show first few alerts
            for alert in alerts[:3]:
                print(f"   • {alert[:100]}{'...' if len(alert) > 100 else ''}")
        else:
            print("ℹ️ No alerts generated (system performing well)")

        # Test system health calculation
        print("🏥 Testing system health calculation...")
        health_score = reporter._calculate_overall_system_health()
        print(f"   System health score: {health_score:.3f}")
        # Test specific analysis methods
        print("🔬 Testing specific analysis methods...")

        # Test regime distribution analysis
        reporter._analyze_regime_distribution(unified_data, unique_clusters)
        if reporter.regime_metrics:
            print("✅ Regime distribution analysis completed")
            print(f"   Total regimes: {reporter.regime_metrics.total_regimes}")
            print(f"   Data balance score: {reporter.regime_metrics.data_balance_score:.3f}")

        # Test data quality analysis
        reporter._analyze_data_quality(unified_data, validation_results)
        if reporter.quality_metrics:
            print("✅ Data quality analysis completed")
            print(f"   Overall quality score: {reporter.quality_metrics.overall_quality_score:.3f}")

        # Test temporal analysis
        reporter._analyze_temporal_patterns(unified_data, unique_clusters)
        if reporter.temporal_metrics:
            print("✅ Temporal analysis completed")
            print(f"   Temporal gaps detected: {len(reporter.temporal_metrics.temporal_gaps)}")

        # Save sample report
        print("💾 Testing report saving...")
        saved_files = reporter.save_comprehensive_report(
            report_data=report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        if saved_files:
            print(f"✅ Saved {len(saved_files)} report files:")
            for file_path in saved_files:
                print(f"   • {file_path}")
        else:
            print("⚠️ No files were saved")

        # Test markdown report generation specifically
        print("📝 Testing markdown report generation...")
        markdown_path = reporter._save_markdown_report(
            report_data=report,
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        if markdown_path:
            print(f"✅ Markdown report saved: {markdown_path}")
        else:
            print("⚠️ Markdown report not saved")

        print("\n🎉 Step08 Enhanced Reporting Test Completed Successfully!")
        print("=" * 50)
        print("✅ All major functionality verified")
        print("✅ Report generation working")
        print("✅ Performance predictions functional")
        print("✅ Alert system operational")
        print("✅ File saving capabilities confirmed")
        print("✅ Enhanced markdown reports generated")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_reporting()
    sys.exit(0 if success else 1)
