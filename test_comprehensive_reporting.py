#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive clustering reporting capabilities.
This script shows how the new reporting system provides in-depth analysis of clustering results.
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.clusters.step10_comprehensive_reporting import (
    ComprehensiveReporter, ComprehensiveReport, ClusterStatistics
)
from src.training.steps.market_analysis.clusters.step1_feature_preparation import ClusteringContext

def create_synthetic_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    print("📊 Creating synthetic market data...")
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_samples//24)  # Assuming hourly data
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate price data with different regimes
    np.random.seed(42)
    prices = [100.0]
    volumes = []
    
    # Create 3 different market regimes
    regime_length = n_samples // 3
    
    for i in range(n_samples):
        if i < regime_length:
            # Bull market regime
            price_change = np.random.normal(0.001, 0.02)  # Positive trend, low volatility
        elif i < 2 * regime_length:
            # Bear market regime
            price_change = np.random.normal(-0.0005, 0.03)  # Negative trend, high volatility
        else:
            # Sideways market regime
            price_change = np.random.normal(0.0, 0.015)  # No trend, medium volatility
        
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        # Generate volume
        volume = np.random.uniform(1000, 10000)
        volumes.append(volume)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    
    print(f"✅ Created synthetic market data with {len(data)} samples")
    return data

def create_synthetic_clustering_result(n_samples: int = 1000, n_clusters: int = 3) -> dict:
    """Create synthetic clustering result for testing."""
    print(f"🔍 Creating synthetic clustering result with {n_clusters} clusters...")
    
    # Generate cluster assignments with some persistence
    np.random.seed(42)
    assignments = []
    current_cluster = 0
    
    for i in range(n_samples):
        # Add some persistence (clusters tend to stay the same)
        if np.random.random() < 0.8:  # 80% chance to stay in same cluster
            assignments.append(current_cluster)
        else:
            current_cluster = np.random.randint(0, n_clusters)
            assignments.append(current_cluster)
    
    # Create clustering result
    result = {
        'cluster_assignments': np.array(assignments),
        'n_clusters': n_clusters,
        'silhouette_score': np.random.uniform(0.3, 0.7),
        'davies_bouldin_score': np.random.uniform(1.5, 3.0),
        'calinski_harabasz_score': np.random.uniform(100, 500),
        'clustering_method': 'advanced_3_step_iterative',
        'risk_mitigation_enabled': True
    }
    
    print(f"✅ Created synthetic clustering result")
    return result

async def test_comprehensive_reporting():
    """Test the comprehensive reporting system."""
    print("🧪 Testing Comprehensive Clustering Reporting")
    print("=" * 60)
    
    try:
        # Create test data
        market_data = create_synthetic_market_data(1000)
        clustering_result = create_synthetic_clustering_result(1000, 3)
        
        # Create clustering context
        features = np.random.randn(1000, 10)  # 10 features
        context = ClusteringContext(
            original_features=features,
            market_data=market_data,
            original_feature_names=[f"feature_{i}" for i in range(10)],
            feature_scores={f"feature_{i}": np.random.random() for i in range(10)}
        )
        
        # Initialize comprehensive reporter
        print("\n📊 Initializing Comprehensive Reporter...")
        reporter = ComprehensiveReporter(verbose=True)
        
        # Generate comprehensive report
        print("\n🔍 Generating comprehensive report...")
        report = reporter.generate_comprehensive_report(
            context=context,
            clustering_result=clustering_result,
            market_data=market_data,
            test_size=0.3,
            n_splits=5
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE CLUSTERING REPORT")
        print("=" * 80)
        
        # Cluster Statistics
        print("\n📈 CLUSTER STATISTICS:")
        print("-" * 40)
        for stat in report.cluster_statistics:
            print(f"Cluster {stat.cluster_id}:")
            print(f"  Size: {stat.size} ({stat.percentage:.1f}%)")
            print(f"  Mean Volatility: {stat.mean_volatility:.4f}")
            print(f"  Mean Return: {stat.mean_return:.4f}")
            print(f"  Sharpe Ratio: {stat.sharpe_ratio:.4f}")
            print(f"  Max Drawdown: {stat.max_drawdown:.4f}")
            print(f"  Persistence Score: {stat.persistence_score:.4f}")
            print(f"  Economic Score: {stat.economic_score:.4f}")
            print()
        
        # Economic Distinctiveness
        print("💰 ECONOMIC DISTINCTIVENESS:")
        print("-" * 40)
        print(f"Volatility Separation Tests: {len(report.economic_distinctiveness.volatility_separation)}")
        print(f"Return Difference Tests: {len(report.economic_distinctiveness.return_differences)}")
        print(f"Sharpe Difference Tests: {len(report.economic_distinctiveness.sharpe_differences)}")
        print(f"Drawdown Hazard Analysis: {len(report.economic_distinctiveness.drawdown_hazard)}")
        print(f"Effect Sizes Calculated: {len(report.economic_distinctiveness.effect_sizes)}")
        print(f"FDR-Corrected P-values: {len(report.economic_distinctiveness.fdr_corrected_pvalues)}")
        
        # Persistence Analysis
        print("\n🔄 REGIME PERSISTENCE ANALYSIS:")
        print("-" * 40)
        print(f"Survival Curves: {list(report.persistence_analysis.survival_curves.keys())}")
        print(f"Transition Matrix Shape: {report.persistence_analysis.transition_matrix.shape}")
        print(f"Stability Metrics: {list(report.persistence_analysis.stability_metrics.keys())}")
        print(f"Horizon Analysis: {list(report.persistence_analysis.horizon_analysis.keys())}")
        
        # In-Sample Metrics
        print("\n📊 IN-SAMPLE METRICS:")
        print("-" * 40)
        for key, value in report.in_sample_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Out-of-Sample Metrics
        print("\n🔍 OUT-OF-SAMPLE METRICS:")
        print("-" * 40)
        for key, value in report.out_of_sample_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Summary Statistics
        print("\n📋 SUMMARY STATISTICS:")
        print("-" * 40)
        for key, value in report.summary_statistics.items():
            print(f"{key}: {value}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"{i}. {recommendation}")
        
        # Save report
        print("\n💾 Saving comprehensive report...")
        output_path = "/tmp/comprehensive_clustering_report.json"
        reporter.save_report(report, output_path)
        print(f"✅ Report saved to {output_path}")
        
        print("\n🎉 Comprehensive reporting test completed successfully!")
        print("✅ The system provides:")
        print("   - Detailed cluster statistics and distribution")
        print("   - Economic distinctiveness analysis with statistical tests")
        print("   - Regime persistence and transition analysis")
        print("   - In-sample and out-of-sample validation")
        print("   - Comprehensive recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_statistical_analysis():
    """Test the statistical analysis capabilities."""
    print("\n🧪 Testing Statistical Analysis Capabilities")
    print("=" * 60)
    
    try:
        # Create test data with known differences
        np.random.seed(42)
        n_samples = 500
        
        # Create two distinct clusters
        cluster1_returns = np.random.normal(0.001, 0.02, n_samples//2)  # Bull market
        cluster2_returns = np.random.normal(-0.0005, 0.03, n_samples//2)  # Bear market
        
        returns = np.concatenate([cluster1_returns, cluster2_returns])
        volatility = np.abs(returns) + np.random.uniform(0.01, 0.05, n_samples)
        assignments = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)]).astype(int)
        
        # Create market data
        market_data = pd.DataFrame({
            'close': 100 * np.cumprod(1 + returns),
            'volume': np.random.uniform(1000, 10000, n_samples)
        })
        
        # Test statistical functions
        reporter = ComprehensiveReporter(verbose=True)
        
        print("🔍 Testing Levene's tests for volatility separation...")
        volatility_separation = reporter._levene_tests_by_cluster(
            pd.Series(volatility), assignments
        )
        print(f"Volatility separation tests: {len(volatility_separation)}")
        
        print("🔍 Testing t-tests for return differences...")
        return_differences = reporter._ttest_returns_by_cluster(
            pd.Series(returns), assignments
        )
        print(f"Return difference tests: {len(return_differences)}")
        
        print("🔍 Testing Mann-Whitney tests for Sharpe ratios...")
        sharpe_ratios = returns / volatility
        sharpe_differences = reporter._mannwhitney_sharpe_by_cluster(
            sharpe_ratios, assignments
        )
        print(f"Sharpe difference tests: {len(sharpe_differences)}")
        
        print("🔍 Testing effect size calculations...")
        effect_sizes = reporter._calculate_effect_sizes(
            pd.Series(volatility), pd.Series(returns), sharpe_ratios, assignments
        )
        print(f"Effect sizes calculated: {len(effect_sizes)}")
        
        print("✅ Statistical analysis tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Statistical analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all comprehensive reporting tests."""
    print("🚀 Starting Comprehensive Clustering Reporting Tests")
    print("=" * 80)
    
    # Test comprehensive reporting
    reporting_success = await test_comprehensive_reporting()
    
    # Test statistical analysis
    statistical_success = await test_statistical_analysis()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE REPORTING TEST SUMMARY")
    print("=" * 80)
    print(f"Comprehensive Reporting: {'✅ PASS' if reporting_success else '❌ FAIL'}")
    print(f"Statistical Analysis: {'✅ PASS' if statistical_success else '❌ FAIL'}")
    
    if reporting_success and statistical_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The comprehensive reporting system provides:")
        print("   📊 Detailed cluster statistics and distribution analysis")
        print("   💰 Economic distinctiveness with statistical validation")
        print("   🔄 Regime persistence and transition analysis")
        print("   📈 In-sample and out-of-sample performance metrics")
        print("   🧪 Statistical tests (Levene's, t-tests, Mann-Whitney)")
        print("   📊 Effect size calculations and FDR correction")
        print("   💡 Actionable recommendations based on analysis")
        print("   📋 Comprehensive summary statistics")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review the errors above and fix any issues.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)