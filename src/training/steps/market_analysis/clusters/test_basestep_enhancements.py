"""
Test script for BaseStep enhancements in market analysis steps.

This script tests that the enhanced steps work correctly with BaseStep comprehensive tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep
from src.training.steps.market_analysis.clusters.step2_initial_clustering import InitialClusteringStep
from src.training.steps.market_analysis.clusters.step8_validation import ValidationStep
from src.training.steps.market_analysis.clusters.step9_results_consolidation import ResultsConsolidationStep
from src.training.steps.market_analysis.clusters.step10_comprehensive_reporting import ComprehensiveReporter


def create_test_data():
    """Create test data for the steps."""
    # Create sample market data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 105,
        'low': np.random.randn(n_samples).cumsum() + 95,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples),
        'volatility': np.random.rand(n_samples) * 0.1,
        'returns': np.random.randn(n_samples) * 0.01,
        'drawdown': np.random.randn(n_samples) * 0.05
    })
    
    # Create sample features
    features = np.random.randn(n_samples, n_features)
    
    return market_data, features


async def test_step1_feature_preparation():
    """Test step1 feature preparation with BaseStep tools."""
    print("🧪 Testing Step 1: Feature Preparation...")
    
    try:
        # Create test data
        market_data, features = create_test_data()
        
        # Create step instance
        step = DataDrivenFeaturePreparationStep(verbose=True)
        
        # Create config
        config = {
            'market_data': market_data,
            'pca_components': 10,
            'use_cv_enhancement': False
        }
        
        # Execute step
        result = await step.execute(config)
        
        # Check result
        assert result['success'] == True, "Step 1 should succeed"
        assert 'artifacts' in result, "Result should contain artifacts"
        assert 'metrics' in result, "Result should contain metrics"
        
        print("✅ Step 1: Feature Preparation - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Step 1: Feature Preparation - FAILED: {e}")
        return False


async def test_step2_initial_clustering():
    """Test step2 initial clustering with BaseStep tools."""
    print("🧪 Testing Step 2: Initial Clustering...")
    
    try:
        # Create test data
        market_data, features = create_test_data()
        
        # Create step instance
        step = InitialClusteringStep(verbose=True)
        
        # Create config with context
        config = {
            'context': type('Context', (), {
                'optimized_features': features,
                'market_data': market_data,
                'original_feature_names': [f'feature_{i}' for i in range(features.shape[1])]
            })(),
            'n_regimes': 5
        }
        
        # Execute step
        result = await step.execute(config)
        
        # Check result
        assert result['success'] == True, "Step 2 should succeed"
        assert 'artifacts' in result, "Result should contain artifacts"
        assert 'metrics' in result, "Result should contain metrics"
        
        print("✅ Step 2: Initial Clustering - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Step 2: Initial Clustering - FAILED: {e}")
        return False


async def test_step8_validation():
    """Test step8 validation with BaseStep tools."""
    print("🧪 Testing Step 8: Validation...")
    
    try:
        # Create test data
        market_data, features = create_test_data()
        assignments = np.random.randint(0, 5, len(features))
        
        # Create step instance
        step = ValidationStep(verbose=True)
        
        # Create config with context
        config = {
            'context': type('Context', (), {
                'optimized_features': features,
                'optimized_assignments': assignments,
                'market_data': market_data
            })(),
            'test_size': 0.3,
            'n_splits': 5
        }
        
        # Execute step
        result = await step.execute(config)
        
        # Check result
        assert result['success'] == True, "Step 8 should succeed"
        assert 'artifacts' in result, "Result should contain artifacts"
        assert 'metrics' in result, "Result should contain metrics"
        
        print("✅ Step 8: Validation - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Step 8: Validation - FAILED: {e}")
        return False


async def test_step9_results_consolidation():
    """Test step9 results consolidation with BaseStep tools."""
    print("🧪 Testing Step 9: Results Consolidation...")
    
    try:
        # Create test data
        market_data, features = create_test_data()
        assignments = np.random.randint(0, 5, len(features))
        
        # Create step instance
        step = ResultsConsolidationStep(verbose=True)
        
        # Create config with context
        config = {
            'context': type('Context', (), {
                'optimized_features': features,
                'optimized_assignments': assignments,
                'market_data': market_data,
                'tas_assignments': np.random.randint(0, 3, len(features)),
                'nas_assignments': np.random.randint(0, 2, len(features))
            })(),
            'output_dir': '/tmp'
        }
        
        # Execute step
        result = await step.execute(config)
        
        # Check result
        assert result['success'] == True, "Step 9 should succeed"
        assert 'artifacts' in result, "Result should contain artifacts"
        assert 'metrics' in result, "Result should contain metrics"
        
        print("✅ Step 9: Results Consolidation - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Step 9: Results Consolidation - FAILED: {e}")
        return False


async def test_step10_comprehensive_reporting():
    """Test step10 comprehensive reporting with BaseStep tools."""
    print("🧪 Testing Step 10: Comprehensive Reporting...")
    
    try:
        # Create test data
        market_data, features = create_test_data()
        assignments = np.random.randint(0, 5, len(features))
        
        # Create step instance
        step = ComprehensiveReporter(verbose=True)
        
        # Create context
        context = type('Context', (), {
            'optimized_features': features,
            'optimized_assignments': assignments,
            'market_data': market_data,
            'tas_assignments': np.random.randint(0, 3, len(features)),
            'nas_assignments': np.random.randint(0, 2, len(features))
        })()
        
        # Create clustering result
        clustering_result = {
            'cluster_assignments': assignments,
            'n_clusters': 5,
            'silhouette_score': 0.5
        }
        
        # Generate report
        report = step.generate_comprehensive_report(
            context=context,
            clustering_result=clustering_result,
            market_data=market_data,
            test_size=0.3,
            n_splits=5
        )
        
        # Check report
        assert report is not None, "Report should be generated"
        assert hasattr(report, 'cluster_statistics'), "Report should have cluster statistics"
        assert hasattr(report, 'economic_distinctiveness'), "Report should have economic distinctiveness"
        assert hasattr(report, 'persistence_analysis'), "Report should have persistence analysis"
        
        print("✅ Step 10: Comprehensive Reporting - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Step 10: Comprehensive Reporting - FAILED: {e}")
        return False


async def test_shared_utils():
    """Test shared utilities with BaseStep tools."""
    print("🧪 Testing Shared Utilities...")
    
    try:
        from src.training.steps.market_analysis.clusters.shared_utils import (
            MetricsCalculator, prepare_market_features, FeatureConfig
        )
        
        # Create test data
        market_data, features = create_test_data()
        assignments = np.random.randint(0, 5, len(features))
        
        # Test MetricsCalculator with BaseStep
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(assignments, market_data)
        
        assert 'consensus_score' in metrics, "Metrics should contain consensus score"
        assert 'economic_score' in metrics, "Metrics should contain economic score"
        
        # Test prepare_market_features with BaseStep
        config = FeatureConfig(n_features=10, use_pca=True, pca_components=5)
        result = prepare_market_features(market_data, config)
        
        assert result.features is not None, "Features should be prepared"
        assert len(result.feature_names) > 0, "Feature names should be provided"
        
        print("✅ Shared Utilities - PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Shared Utilities - FAILED: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting BaseStep Enhancement Tests...")
    print("=" * 50)
    
    tests = [
        test_step1_feature_preparation,
        test_step2_initial_clustering,
        test_step8_validation,
        test_step9_results_consolidation,
        test_step10_comprehensive_reporting,
        test_shared_utils
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BaseStep enhancements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)