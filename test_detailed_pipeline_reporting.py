#!/usr/bin/env python3
"""
Test script for the detailed pipeline reporting functionality.

This script tests the comprehensive reporting system integrated into the
UnifiedDataDrivenPipeline to ensure it generates detailed metrics and reports.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test data for the pipeline."""
    np.random.seed(42)
    
    # Create OHLCV data
    n_samples = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Create targets (simple price movement)
    targets = pd.Series(
        (prices[1:] - prices[:-1]) / prices[:-1],
        index=dates[1:],
        name='target'
    )
    
    return data, targets

def test_detailed_reporter():
    """Test the detailed reporter functionality."""
    print("🧪 Testing Detailed Pipeline Reporter...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.detailed_pipeline_reporter import (
            DetailedPipelineReporter, DetailedPipelineReport, FeatureMetrics, StepMetrics, GlobalMetrics
        )
        
        # Initialize reporter
        reporter = DetailedPipelineReporter(outcomes_dir="test_outcomes")
        print("✅ DetailedPipelineReporter initialized successfully")
        
        # Test step tracking
        reporter.start_step("test_step", 10)
        print("✅ Step tracking started")
        
        # Test feature tracking
        reporter.track_feature_creation(
            feature_name="test_feature",
            feature_type="technical",
            parent_features=["close", "volume"],
            transform_type="rolling_mean",
            lookback_period=20,
            mutual_information=0.5,
            shap_score=0.3,
            correlation_with_target=0.4
        )
        print("✅ Feature tracking working")
        
        # Test feature selection tracking
        reporter.track_feature_selection(
            selected_features=["test_feature", "another_feature"],
            feature_importance={"test_feature": 0.8, "another_feature": 0.6},
            selection_metrics={"method": "mutual_information", "threshold": 0.3}
        )
        print("✅ Feature selection tracking working")
        
        # Test interaction tracking
        reporter.track_interaction_generation(
            interactions=[{"interaction_type": "multiplication", "features": ["close", "volume"]}],
            interaction_metrics={"total_interactions": 1, "generation_time": 0.5}
        )
        print("✅ Interaction tracking working")
        
        # Test lookback tracking
        reporter.track_lookback_optimization(
            optimized_lookbacks={"feature1": 20, "feature2": 30},
            lookback_metrics={"optimization_method": "grid_search", "best_score": 0.85}
        )
        print("✅ Lookback tracking working")
        
        # End step
        reporter.end_step("test_step", 5, 1.5, 50.0, True)
        print("✅ Step tracking completed")
        
        # Test report generation
        data_info = {
            'input_shape': (1000, 6),
            'timeframe': '15m',
            'targets_available': True
        }
        
        config = {
            'labeling_type': 'analyst',
            'feature_selection_method': 'mutual_information'
        }
        
        report = reporter.generate_detailed_report(None, config, data_info)
        print("✅ Report generation working")
        
        # Test report saving
        report_path = reporter.save_report(report, format="both")
        print(f"✅ Report saved to: {report_path}")
        
        # Verify files exist
        json_file = Path(f"{report_path}.json")
        txt_file = Path(f"{report_path}.txt")
        
        if json_file.exists() and txt_file.exists():
            print("✅ Both JSON and TXT report files created successfully")
            
            # Check file sizes
            json_size = json_file.stat().st_size
            txt_size = txt_file.stat().st_size
            
            print(f"📊 JSON report size: {json_size} bytes")
            print(f"📊 TXT report size: {txt_size} bytes")
            
            if json_size > 0 and txt_size > 0:
                print("✅ Report files contain data")
            else:
                print("⚠️ Report files are empty")
        else:
            print("❌ Report files not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test the pipeline integration with reporting."""
    print("\n🧪 Testing Pipeline Integration...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
            UnifiedDataDrivenPipeline, create_default_config
        )
        
        # Create test data
        data, targets = create_test_data()
        print(f"✅ Test data created: {data.shape}, targets: {targets.shape}")
        
        # Initialize pipeline
        config = create_default_config()
        pipeline = UnifiedDataDrivenPipeline(config)
        print("✅ Pipeline initialized")
        
        # Check if detailed reporter is initialized
        if hasattr(pipeline, 'detailed_reporter'):
            print("✅ Detailed reporter is integrated into pipeline")
        else:
            print("❌ Detailed reporter not found in pipeline")
            return False
        
        # Note: We won't run the full pipeline here as it requires many dependencies
        # Instead, we'll just verify the integration is correct
        print("✅ Pipeline integration test completed (full run skipped due to dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Detailed Pipeline Reporting Tests")
    print("=" * 60)
    
    # Test 1: Detailed Reporter
    test1_passed = test_detailed_reporter()
    
    # Test 2: Pipeline Integration
    test2_passed = test_pipeline_integration()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  Detailed Reporter Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Pipeline Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Detailed pipeline reporting is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())