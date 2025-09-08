#!/usr/bin/env python3
"""
Demo script for Step05 Enhanced Labeling Reporting

This script demonstrates the functionality of the Step05EnhancedReporter
with sample labeled data and comprehensive labeling analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.step05_enhanced_reporting import Step05EnhancedReporter
import collections
import time

def generate_sample_labeled_data() -> pd.DataFrame:
    """Generate sample labeled market data for demonstration."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(2000)]

    # Generate synthetic price data with regime characteristics
    base_price = 50000
    prices = []
    labels = []
    meta_labels = []
    confidences = []
    current_price = base_price

    for i in range(2000):
        # Simulate different market conditions
        if i < 500:  # Bull market
            trend = 0.0002
            volatility = 0.008
            label_prob = [0.6, 0.3, 0.1]  # Buy, Hold, Sell
        elif i < 1000:  # Bear market
            trend = -0.00015
            volatility = 0.012
            label_prob = [0.1, 0.3, 0.6]  # Buy, Hold, Sell
        elif i < 1500:  # Sideways market
            trend = 0.00005
            volatility = 0.006
            label_prob = [0.3, 0.4, 0.3]  # Buy, Hold, Sell
        else:  # High volatility
            trend = 0.0001
            volatility = 0.018
            label_prob = [0.25, 0.25, 0.5]  # Buy, Hold, Sell

        change = np.random.normal(trend, volatility)
        current_price *= (1 + change)
        prices.append(current_price)

        # Generate labels based on probabilities
        label = np.random.choice([-1, 0, 1], p=label_prob)  # -1=Sell, 0=Hold, 1=Buy
        labels.append(label)

        # Generate meta-labels (simplified - in practice would be more sophisticated)
        if label == 1:  # Buy signal
            meta_label = np.random.choice([1, 0], p=[0.8, 0.2])  # 80% agreement
        elif label == -1:  # Sell signal
            meta_label = np.random.choice([-1, 0], p=[0.85, 0.15])  # 85% agreement
        else:  # Hold signal
            meta_label = np.random.choice([0, 1, -1], p=[0.7, 0.15, 0.15])  # 70% agreement
        meta_labels.append(meta_label)

        # Generate confidence scores
        if label == meta_label:
            confidence = np.random.uniform(0.7, 0.95)  # High confidence for agreement
        else:
            confidence = np.random.uniform(0.3, 0.7)   # Lower confidence for disagreement
        confidences.append(confidence)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, 2000),
        'label': labels,
        'meta_label': meta_labels,
        'confidence': confidences,
        'regime_id': [0] * 500 + [1] * 500 + [2] * 500 + [3] * 500  # Different regimes
    })

    df.set_index('timestamp', inplace=True)
    return df

def generate_sample_labeling_results() -> dict:
    """Generate sample labeling results."""
    return {
        'total_labels': 2000,
        'label_distribution': {'buy': 650, 'hold': 600, 'sell': 750},
        'labeling_method': 'meta_labeling',
        'processing_time': 45.67,
        'success': True,
        'label_quality_score': 0.87,
        'meta_label_agreement': 0.82
    }

def generate_sample_performance_data() -> dict:
    """Generate sample performance data."""
    return {
        'execution_time': 45.67,
        'memory_usage': 256.8,
        'cpu_usage': 42.3,
        'label_creation_rate': 43.8,  # labels/second
        'meta_labeling_time': 32.4,
        'fallback_labeling_time': 8.2,
        'validation_time': 5.1,
        'total_function_calls': 1250,
        'successful_operations': 1245,
        'failed_operations': 5,
        'error_rate': 0.004,
        'processing_efficiency': 0.89,
        'optimization_effectiveness': 0.94
    }

def generate_sample_validation_results() -> dict:
    """Generate sample validation results."""
    return {
        'passed': True,
        'checks_performed': 8,
        'failures': 0,
        'error_rate': 0.0,
        'data_integrity_score': 0.96,
        'label_consistency_score': 0.91,
        'statistical_validation_score': 0.88,
        'cross_validation_score': 0.85,
        'warnings': ['Minor label distribution imbalance detected'],
        'recommendations': [
            'Labels generated successfully with high quality',
            'Consider additional meta-labeling for edge cases',
            'Label distribution is slightly skewed towards sell signals'
        ]
    }

def generate_sample_meta_labeling_analysis() -> dict:
    """Generate sample meta-labeling analysis."""
    return {
        'meta_labels_created': 2000,
        'success_rate': 0.94,
        'avg_confidence': 0.79,
        'quality_score': 0.86,
        'agreement_rate': 0.82,
        'computation_time': 32.4,
        'memory_usage': 184.5,
        'optimization_gain': 0.18
    }

def demo_step05_labeling():
    """Demonstrate Step05 enhanced labeling reporting."""
    print("🏷️ Demonstrating Step05 Enhanced Labeling Reporting")
    print("=" * 70)

    try:
        # Initialize the enhanced reporter
        print("📊 Initializing Step05 Enhanced Reporter...")
        reporter = Step05EnhancedReporter()
        print("✅ Reporter initialized successfully")

        # Generate sample data
        print("📈 Generating sample labeled data...")
        labeled_data = generate_sample_labeled_data()
        print(f"✅ Generated {len(labeled_data)} labeled data points")

        # Generate sample results
        print("🔍 Generating sample labeling analysis...")
        labeling_results = generate_sample_labeling_results()
        performance_data = generate_sample_performance_data()
        validation_results = generate_sample_validation_results()
        meta_labeling_analysis = generate_sample_meta_labeling_analysis()
        print("✅ Sample analysis data generated")

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            labeled_data=labeled_data,
            labeling_results=labeling_results,
            performance_data=performance_data,
            validation_results=validation_results,
            meta_labeling_analysis=meta_labeling_analysis,
            symbol="BTCUSDT",
            exchange="BINANCE",
            timeframe="1h"
        )
        print("✅ Comprehensive report generated")

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report=comprehensive_report,
            base_filename="demo_step05_enhanced_labeling"
        )
        print("✅ Report saved successfully")
        print(f"📁 Files saved: {saved_files}")

        return True

    except Exception as e:
        print(f"❌ Step05 demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function."""
    print("🚀 Starting Step05 Enhanced Labeling Demo")
    print("=" * 50)

    # Demo the enhanced labeling reporting
    success = demo_step05_labeling()

    # Display summary
    print("\n" + "=" * 70)
    print("🎯 DEMO SUMMARY")
    print("=" * 70)
    print(f"📊 Reports Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📂 Step Demonstrated:")
    print("   • Step05: Enhanced Labeling with Meta-Labeling")
    print("📁 Output Directories:")
    print("   • src/training/reports/step05/")
    print("🔧 Enhanced Features:")
    print("   • Label quality assessment and validation")
    print("   • Meta-labeling performance analysis")
    print("   • Trading strategy implications")
    print("   • Label distribution analysis")
    print("   • Performance optimization recommendations")
    print("   • Multiple output formats (JSON, Markdown, CSV, Visualizations)")

    if success:
        print("\n✅ Demo completed successfully!")
        print("📋 Check the generated reports for detailed labeling analysis!")
    else:
        print("\n⚠️ Demo encountered issues - check logs above!")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
