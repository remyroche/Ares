#!/usr/bin/env python3
"""
Demo script for Step04 Enhanced Reporting System

This script demonstrates the functionality of the Step04EnhancedReporter
with sample regime data splitting and triple barrier method results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.step04_enhanced_reporting import Step04EnhancedReporter

def generate_sample_regime_data() -> pd.DataFrame:
    """Generate sample regime-labeled market data for demonstration."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(2000)]

    # Generate synthetic price data with regime characteristics
    base_price = 50000
    prices = []
    regimes = []
    current_price = base_price

    for i in range(2000):
        # Simulate different regimes with different characteristics
        if i < 500:  # Bull regime
            regime = 0
            trend = 0.0002
            volatility = 0.008
        elif i < 1000:  # Bear regime
            regime = 1
            trend = -0.00015
            volatility = 0.012
        elif i < 1500:  # Sideways regime
            regime = 2
            trend = 0.00005
            volatility = 0.006
        else:  # High volatility regime
            regime = 3
            trend = 0.0001
            volatility = 0.018

        change = np.random.normal(trend, volatility)
        current_price *= (1 + change)
        prices.append(current_price)
        regimes.append(regime)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, 2000),
        'regime_id': regimes
    })

    df.set_index('timestamp', inplace=True)
    return df

def generate_sample_labeled_data() -> pd.DataFrame:
    """Generate sample triple barrier labeled data."""
    np.random.seed(123)  # Different seed for variety

    # Generate base data
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(1000)]

    # Generate price data
    base_price = 45000
    prices = []
    current_price = base_price

    for i in range(1000):
        change = np.random.normal(0.0001, 0.01)
        current_price *= (1 + change)
        prices.append(current_price)

    # Generate triple barrier labels (simplified)
    labels = []
    for i in range(1000):
        rand = np.random.random()
        if rand < 0.35:  # 35% profit target hit
            labels.append(1)
        elif rand < 0.65:  # 30% stop loss hit
            labels.append(-1)
        else:  # 35% timeout
            labels.append(0)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, 1000),
        'label': labels
    })

    df.set_index('timestamp', inplace=True)
    return df

def generate_sample_data_splitting_results() -> dict:
    """Generate sample data splitting results."""
    return {
        'success': True,
        'total_regimes': 4,
        'regime_ids': [0, 1, 2, 3],
        'data_shape': (2000, 7),
        'processing_method': 'streaming',
        'memory_usage': {'peak_mb': 512, 'avg_mb': 256},
        'total_samples': 2000,
        'samples_per_regime': [500, 500, 500, 500],
        'regime_balance_score': 95.2,
        'data_retention_rate': 98.5,
        'duplicate_handling_efficiency': 99.8
    }

def generate_sample_triple_barrier_results() -> dict:
    """Generate sample triple barrier method results."""
    return {
        'success': True,
        'total_signals': 1000,
        'signal_distribution': {'buy': 350, 'sell': 300, 'hold': 350},
        'profit_targets_hit': 350,
        'stop_losses_hit': 300,
        'timeouts': 350,
        'avg_profit_target': 0.02,
        'avg_stop_loss': 0.015,
        'avg_timeout_days': 5,
        'signal_confidence': 0.82,
        'signal_purity': 0.76,
        'false_signal_rate': 0.14,
        'effectiveness_score': 0.79,
        'signals': {i: np.random.choice([-1, 0, 1]) for i in range(1000)}
    }

def generate_sample_performance_data(step_type: str = "regime_data_splitting") -> dict:
    """Generate sample performance data."""
    if step_type == "regime_data_splitting":
        return {
            'execution_time': 145.67,
            'memory_usage': 384.5,
            'cpu_usage': 45.2,
            'data_processing_rate': 13750.0,  # rows/second
            'file_processing_rate': 2.5,       # files/second
            'merging_time': 45.2,
            'splitting_time': 89.4,
            'validation_time': 11.1,
            'total_function_calls': 850,
            'successful_operations': 845,
            'failed_operations': 5,
            'error_rate': 0.006,
            'data_retention_rate': 0.985,
            'duplicate_handling_efficiency': 0.998
        }
    else:  # triple_barrier_method
        return {
            'execution_time': 89.34,
            'memory_usage': 256.8,
            'cpu_usage': 38.7,
            'signal_generation_rate': 11200.0,  # signals/second
            'label_creation_time': 71.5,
            'barrier_calculation_time': 13.4,
            'validation_time': 4.4,
            'total_signals_generated': 1000,
            'successful_labels': 987,
            'failed_labels': 13,
            'label_success_rate': 0.987,
            'profit_target_achieved': 350,
            'stop_loss_hit': 300,
            'timeout_reached': 350
        }

def demo_regime_data_splitting():
    """Demonstrate Step04 regime data splitting reporting."""
    print("🔀 Demonstrating Step04 Regime Data Splitting Enhanced Reporting")
    print("=" * 70)

    try:
        # Initialize the enhanced reporter
        print("📊 Initializing Step04 Enhanced Reporter...")
        reporter = Step04EnhancedReporter()
        print("✅ Reporter initialized successfully")

        # Generate sample data
        print("📈 Generating sample regime-labeled data...")
        regime_data = generate_sample_regime_data()
        print(f"✅ Generated {len(regime_data)} data points with {regime_data['regime_id'].nunique()} regimes")

        # Generate sample results
        print("🔍 Generating sample data splitting results...")
        data_splitting_results = generate_sample_data_splitting_results()
        performance_data = generate_sample_performance_data("regime_data_splitting")
        print("✅ Sample results generated")

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            data_splitting_results=data_splitting_results,
            triple_barrier_results={},  # No triple barrier results
            regime_data=regime_data,
            performance_data=performance_data,
            symbol="BTCUSDT",
            exchange="BINANCE",
            timeframe="1h",
            step_type="regime_data_splitting"
        )
        print("✅ Comprehensive report generated")

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report=comprehensive_report,
            base_filename="demo_step04_regime_splitting"
        )
        print("✅ Report saved successfully")
        print(f"📁 Files saved: {saved_files}")

        return True

    except Exception as e:
        print(f"❌ Regime data splitting demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_triple_barrier_method():
    """Demonstrate Step04_5 triple barrier method reporting."""
    print("\n🎯 Demonstrating Step04_5 Triple Barrier Method Enhanced Reporting")
    print("=" * 70)

    try:
        # Initialize the enhanced reporter
        print("📊 Initializing Step04 Enhanced Reporter...")
        reporter = Step04EnhancedReporter()
        print("✅ Reporter initialized successfully")

        # Generate sample data
        print("📈 Generating sample labeled data...")
        labeled_data = generate_sample_labeled_data()
        regime_data = generate_sample_regime_data().head(1000)  # Match the labeled data size
        print(f"✅ Generated {len(labeled_data)} labeled data points")

        # Generate sample results
        print("🔍 Generating sample triple barrier results...")
        triple_barrier_results = generate_sample_triple_barrier_results()
        performance_data = generate_sample_performance_data("triple_barrier_method")
        print("✅ Sample results generated")

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            data_splitting_results={},  # No data splitting results
            triple_barrier_results=triple_barrier_results,
            regime_data=regime_data,
            performance_data=performance_data,
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1h",
            step_type="triple_barrier_method"
        )
        print("✅ Comprehensive report generated")

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report=comprehensive_report,
            base_filename="demo_step04_triple_barrier"
        )
        print("✅ Report saved successfully")
        print(f"📁 Files saved: {saved_files}")

        return True

    except Exception as e:
        print(f"❌ Triple barrier method demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function."""
    print("🚀 Starting Step04 Enhanced Reporting Demo")
    print("=" * 50)

    # Demo both step types
    success1 = demo_regime_data_splitting()
    success2 = demo_triple_barrier_method()

    # Display summary
    print("\n" + "=" * 70)
    print("🎯 DEMO SUMMARY")
    print("=" * 70)
    print(f"📊 Reports Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📂 Report Types Demonstrated:")
    print("   • Step04: Regime Data Splitting")
    print("   • Step04_5: Triple Barrier Method")
    print("📁 Output Directories:")
    print("   • src/training/reports/step04/")
    print("🔧 Enhanced Features:")
    print("   • Comprehensive performance metrics")
    print("   • Data quality assessment")
    print("   • Regime analysis")
    print("   • Trading signal analysis")
    print("   • Multiple output formats (JSON, Markdown, CSV, Visualizations)")

    if success1 and success2:
        print("\n✅ All demos completed successfully!")
        print("📋 Check the generated reports for detailed analysis!")
    else:
        print("\n⚠️ Some demos encountered issues - check logs above!")

    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
