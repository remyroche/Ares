#!/usr/bin/env python3
"""
Demo script for Step03 Enhanced Reporting System

This script demonstrates the functionality of the Step03EnhancedReporter
with sample market data and regime analysis results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.hmm_clustering.step03_enhanced_reporting import Step03EnhancedReporter
import time

def generate_sample_market_data() -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    np.random.seed(42)  # For reproducible results

    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(1000)]

    # Generate synthetic price data with some regime-like behavior
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(1000):
        # Add some trend and volatility changes
        trend = 0.0001 if i < 300 else (-0.00005 if i < 600 else 0.0002)
        volatility = 0.01 if i < 300 else (0.005 if i < 600 else 0.015)

        change = np.random.normal(trend, volatility)
        current_price *= (1 + change)
        prices.append(current_price)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, 1000)
    })

    df.set_index('timestamp', inplace=True)
    return df

def generate_sample_hmm_results() -> dict:
    """Generate sample HMM analysis results."""
    return {
        'n_components': 3,
        'log_likelihood': -1250.5,
        'converged': True,
        'covariance_type': 'full',
        'model_type': 'GaussianHMM',
        'transition_matrix': [
            [0.85, 0.10, 0.05],
            [0.15, 0.80, 0.05],
            [0.10, 0.20, 0.70]
        ],
        'steady_state_probabilities': [0.4, 0.35, 0.25],
        'feature_importance': {
            'returns': 0.35,
            'volatility': 0.28,
            'volume': 0.20,
            'momentum': 0.17
        },
        'regime_persistence': [15.2, 12.8, 18.5],  # days
        'volatility_by_regime': [0.012, 0.008, 0.018],  # daily volatility
        'trend_by_regime': [0.0005, -0.0002, 0.0012],  # daily trend
        'regime_confidence': [0.82, 0.75, 0.88]
    }

def generate_sample_clustering_results() -> dict:
    """Generate sample clustering analysis results."""
    return {
        'silhouette_score': 0.65,
        'davies_bouldin': 0.78,
        'calinski_harabasz': 1245.5,
        'n_clusters': 3,
        'cluster_sizes': [320, 280, 400],
        'cluster_centers': [
            [0.001, 0.012, 1000000],
            [-0.0005, 0.008, 800000],
            [0.002, 0.018, 1200000]
        ],
        'explained_variance': 0.82,
        'reduction_efficiency': 0.91,
        'stability_score': 0.76
    }

def generate_sample_performance_data() -> dict:
    """Generate sample performance data."""
    return {
        'execution_time': 245.67,
        'memory_usage': 1250.5,
        'cpu_usage': 68.3,
        'processing_rate': 4.08,
        'hmm_training_time': 89.2,
        'clustering_time': 45.8,
        'regime_analysis_time': 67.4,
        'report_generation_time': 12.3,
        'total_function_calls': 1250,
        'successful_operations': 1245,
        'failed_operations': 5,
        'error_rate': 0.004,
        'convergence_iterations': 42,
        'log_likelihood': -1250.5
    }

def main():
    """Main demonstration function."""
    print("🚀 Starting Step03 Enhanced Reporting Demo")
    print("=" * 50)

    try:
        # Initialize the enhanced reporter
        print("📊 Initializing Step03 Enhanced Reporter...")
        reporter = Step03EnhancedReporter()
        print("✅ Reporter initialized successfully")

        # Generate sample data
        print("📈 Generating sample market data...")
        market_data = generate_sample_market_data()
        print(f"✅ Generated {len(market_data)} data points")

        # Generate sample analysis results
        print("🔍 Generating sample HMM and clustering results...")
        hmm_results = generate_sample_hmm_results()
        clustering_results = generate_sample_clustering_results()
        performance_data = generate_sample_performance_data()
        print("✅ Sample results generated")

        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        comprehensive_report = reporter.generate_comprehensive_report(
            hmm_results=hmm_results,
            clustering_results=clustering_results,
            performance_data=performance_data,
            market_data=market_data,
            symbol="BTCUSDT",
            exchange="BINANCE",
            timeframe="1h"
        )
        print("✅ Comprehensive report generated")

        # Save the report
        print("💾 Saving comprehensive report...")
        saved_files = reporter.save_comprehensive_report(
            report=comprehensive_report,
            base_filename="demo_step03_enhanced_report"
        )
        print("✅ Report saved successfully")
        print(f"📁 Files saved: {saved_files}")

        # Display summary
        print("\n" + "=" * 50)
        print("🎯 DEMO SUMMARY")
        print("=" * 50)
        print(f"📊 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 Market Data Points: {len(market_data)}")
        print(f"🔄 HMM Components: {hmm_results['n_components']}")
        print(f"📊 Clusters Found: {clustering_results['n_clusters']}")
        print(f"⚡ Execution Time: {performance_data['execution_time']:.2f}s")
        print(f"💾 Files Created: {len(saved_files)}")
        print("\n📂 Output files:")
        for file_type, file_path in saved_files.items():
            print(f"   • {file_type.upper()}: {file_path}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
