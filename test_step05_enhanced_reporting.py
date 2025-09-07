#!/usr/bin/env python3
"""
Test script for enhanced Step05 reporting system.

This script demonstrates the comprehensive reporting capabilities
of the enhanced Step05 labeling system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from src.training.steps.step05_enhanced_reporting import Step05EnhancedReporter


def create_sample_labeled_data():
    """Create sample labeled data for testing."""
    np.random.seed(42)

    # Generate timestamps
    base_time = datetime.now()
    timestamps = [base_time + timedelta(minutes=i) for i in range(1000)]

    # Generate sample price data
    prices = []
    price = 50000.0
    for i in range(1000):
        # Add some trend and noise
        trend = 0.001 * np.sin(i / 50)  # Slow trend
        noise = np.random.normal(0, 0.005)  # Random noise
        price *= (1 + trend + noise)
        prices.append(price)

    # Generate labels based on price movements
    labels = []
    for i in range(1000):
        if i < 5:
            labels.append(0)  # Hold for first few
            continue

        # Look at recent price changes
        recent_prices = prices[max(0, i-10):i+1]
        if len(recent_prices) < 5:
            labels.append(0)
            continue

        # Simple labeling logic
        short_ma = np.mean(recent_prices[-5:])
        long_ma = np.mean(recent_prices[-20:])

        if short_ma > long_ma * 1.002:  # Bullish
            labels.append(1)
        elif short_ma < long_ma * 0.998:  # Bearish
            labels.append(-1)
        else:  # Neutral
            labels.append(0)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'label': labels,
        'label_confidence': np.random.uniform(0.5, 0.95, 1000),
        'regime_id': np.random.choice([1, 2, 3], 1000, p=[0.4, 0.4, 0.2])
    })

    return df


def create_sample_performance_data():
    """Create sample performance data."""
    return {
        'execution_time': 45.67,
        'memory_usage': 256.8,
        'cpu_usage': 78.5,
        'label_creation_rate': 850.0,
        'meta_labeling_time': 12.3,
        'fallback_labeling_time': 2.1,
        'validation_time': 8.9,
        'function_calls': 1500,
        'successful_ops': 1485,
        'failed_ops': 15,
        'error_rate': 0.01,
        'processing_efficiency': 0.87,
        'optimization_effectiveness': 0.92
    }


def create_sample_validation_results():
    """Create sample validation results."""
    return {
        'passed': True,
        'checks_performed': 25,
        'failures': 1,
        'error_rate': 0.04,
        'data_integrity_score': 0.96,
        'label_consistency_score': 0.89,
        'statistical_score': 0.91,
        'cross_validation_score': 0.88,
        'warnings': [
            'Minor inconsistency in label distribution detected',
            'Some labels may benefit from additional validation'
        ],
        'recommendations': [
            'Consider implementing additional cross-validation checks',
            'Review labeling thresholds for edge cases'
        ]
    }


def create_sample_meta_labeling_analysis():
    """Create sample meta-labeling analysis."""
    return {
        'meta_labels_created': 980,
        'success_rate': 0.94,
        'avg_confidence': 0.82,
        'quality_score': 0.88,
        'agreement_rate': 0.91,
        'computation_time': 15.2,
        'memory_usage': 189.5,
        'optimization_gain': 1.8
    }


def create_sample_labeling_results():
    """Create sample labeling results."""
    return {
        'total_labels_processed': 1000,
        'labels_by_type': {'buy': 324, 'sell': 298, 'hold': 378},
        'confidence_distribution': {'high': 0.65, 'medium': 0.25, 'low': 0.10},
        'processing_stats': {
            'avg_processing_time_per_label': 0.045,
            'peak_memory_usage': 280.5,
            'cpu_utilization': 82.3
        }
    }


def main():
    """Main test function."""
    print("🧪 Testing Enhanced Step05 Reporting System")
    print("=" * 60)

    try:
        # Create sample data
        print("📊 Creating sample labeled data...")
        labeled_data = create_sample_labeled_data()

        # Initialize reporter
        print("📋 Initializing enhanced reporter...")
        reporter = Step05EnhancedReporter()

        # Create sample inputs
        performance_data = create_sample_performance_data()
        validation_results = create_sample_validation_results()
        meta_labeling_analysis = create_sample_meta_labeling_analysis()
        labeling_results = create_sample_labeling_results()

        # Generate comprehensive report
        print("🔍 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            labeled_data=labeled_data,
            labeling_results=labeling_results,
            performance_data=performance_data,
            validation_results=validation_results,
            meta_labeling_analysis=meta_labeling_analysis,
            symbol='BTCUSDT',
            exchange='binance',
            timeframe='5m'
        )

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report=report,
            base_filename="test_step05_enhanced_report"
        )

        print("✅ Enhanced Step05 report generation completed successfully!")
        print("\n📁 Generated Files:")
        for file_type, file_path in saved_files.items():
            if file_path and not file_path.startswith('error'):
                print(f"  - {file_type.upper()}: {file_path}")

        # Display key metrics
        print("\n📊 Key Report Highlights:")
        if 'label_quality_assessment' in report:
            quality = report['label_quality_assessment']
            if 'quality_metrics' in quality:
                metrics = quality['quality_metrics']
                print(f"  - Total Labels: {metrics.get('total_labels', 0):,}")
                print(f"  - Label Confidence: {metrics.get('label_confidence_score', 0):.1f}%")
                print(f"  - Label Consistency: {metrics.get('label_consistency_score', 0):.1f}%")
        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            if 'metrics' in perf:
                metrics = perf['metrics']
                print(f"  - Execution Time: {metrics.get('execution_time_seconds', 0):.2f}s")
                print(f"  - Success Rate: {(metrics.get('successful_operations', 0) / max(1, metrics.get('total_function_calls', 1))) * 100:.1f}%")

        print("\n🎯 Test completed successfully!")
        print("The enhanced Step05 reporting system is now on par with step02_5 and step03_5 reports.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
