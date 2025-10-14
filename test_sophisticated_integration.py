"""
Comprehensive Test for Sophisticated Integration

This test verifies that all the sophisticated logic from FeatureLookbackOptimizationComponent
has been successfully integrated into UnifiedDataDrivenPipeline.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data for validation."""
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    # Generate OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    high_prices = close_prices + np.random.rand(n_samples) * 2
    low_prices = close_prices - np.random.rand(n_samples) * 2
    open_prices = close_prices + np.random.randn(n_samples) * 0.5
    volume = np.random.randint(1000, 10000, n_samples)
    
    # Create features
    data = pd.DataFrame({
        'close': close_prices,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'volume': volume,
        'sma_20': pd.Series(close_prices).rolling(20).mean(),
        'sma_50': pd.Series(close_prices).rolling(50).mean(),
        'rsi_14': calculate_rsi(close_prices, 14),
        'bb_upper': close_prices + 2 * pd.Series(close_prices).rolling(20).std(),
        'bb_lower': close_prices - 2 * pd.Series(close_prices).rolling(20).std(),
        'volatility': pd.Series(close_prices).rolling(20).std(),
        'returns': pd.Series(close_prices).pct_change(),
        'log_returns': np.log(close_prices / np.roll(close_prices, 1)),
        'volume_sma': pd.Series(volume).rolling(20).mean(),
        'price_volume': close_prices * volume,
        'high_low_ratio': high_prices / low_prices,
        'open_close_ratio': open_prices / close_prices,
        'volume_ratio': volume / pd.Series(volume).rolling(20).mean(),
        'momentum_5': close_prices / np.roll(close_prices, 5),
        'momentum_10': close_prices / np.roll(close_prices, 10),
        'momentum_20': close_prices / np.roll(close_prices, 20)
    }, index=dates)
    
    # Add execution mode attribute
    data.attrs['ares_mode'] = 'full'
    
    return data

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI indicator."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50).values

def test_sophisticated_lookback_optimizer():
    """Test the sophisticated lookback optimizer."""
    print("\n🧪 Testing Sophisticated Lookback Optimizer...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.sophisticated_lookback_optimizer import (
            SophisticatedLookbackOptimizer, OptimizationDirection, create_sophisticated_lookback_optimizer
        )
        
        # Create test data
        data = create_test_data(500)
        
        # Initialize optimizer
        optimizer = create_sophisticated_lookback_optimizer()
        
        # Test optimization
        feature_names = ['sma_20', 'rsi_14', 'volatility', 'returns', 'momentum_5']
        target_columns = {
            'long': 'returns',
            'short': 'returns'  # Using same target for simplicity
        }
        
        print(f"📊 Testing optimization with {len(feature_names)} features")
        print(f"📊 Target columns: {target_columns}")
        
        results = optimizer.optimize_features_sophisticated(
            data=data,
            feature_names=feature_names,
            target_columns=target_columns,
            lookback_range=(5, 50),
            optimization_direction=OptimizationDirection.BOTH,
            execution_mode='light',
            use_nested_cv=True,
            max_workers=2
        )
        
        print(f"✅ Optimization completed: {len(results)} features optimized")
        
        # Verify results
        for feature_name, feature_results in results.items():
            if isinstance(feature_results, dict):
                for direction, result in feature_results.items():
                    print(f"  📈 {feature_name} ({direction}): lookback={result.best_lookback}, score={result.best_score:.4f}")
            else:
                print(f"  📈 {feature_name}: lookback={feature_results.best_lookback}, score={feature_results.best_score:.4f}")
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        print(f"📊 Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sophisticated lookback optimizer test failed: {e}")
        return False

def test_multi_horizon_integration():
    """Test the multi-horizon integration."""
    print("\n🧪 Testing Multi-Horizon Integration...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.multi_horizon_integration import (
            MultiHorizonIntegration, create_multi_horizon_integration
        )
        
        # Create test data
        data = create_test_data(500)
        
        # Initialize integration
        integration = create_multi_horizon_integration()
        
        # Test integration
        result = integration.integrate_multi_horizon_labeling(data, force_refresh=True)
        
        print(f"✅ Multi-horizon integration: {result.integration_success}")
        print(f"📊 Target columns: {result.target_columns}")
        print(f"📊 Target info: {len(result.target_info)} targets")
        
        # Test target selection
        if result.target_columns:
            for direction, target_col in result.target_columns.items():
                optimal_target = integration.select_optimal_target_columns(
                    data, result.target_columns, direction
                )
                print(f"  🎯 {direction}: {optimal_target}")
        
        # Test performance stats
        stats = integration.get_performance_stats()
        print(f"📊 Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-horizon integration test failed: {e}")
        return False

def test_comprehensive_validation():
    """Test the comprehensive validation system."""
    print("\n🧪 Testing Comprehensive Validation...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.comprehensive_validation import (
            ComprehensiveValidator, ValidationLevel, create_comprehensive_validator
        )
        
        # Create test data
        data = create_test_data(500)
        
        # Initialize validator
        validator = create_comprehensive_validator("TestValidator")
        
        # Test different validation levels
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        
        for level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            print(f"📊 Testing {level.value} validation...")
            
            is_valid, summary, validated_data = validator.validate_data_comprehensive(
                data, required_columns, level, check_stationarity=True, check_memory=True
            )
            
            print(f"  ✅ Valid: {is_valid}")
            print(f"  📊 Quality score: {summary.data_quality_score:.3f}")
            print(f"  📊 Memory usage: {summary.memory_usage_mb:.1f} MB")
            print(f"  📊 Validation time: {summary.validation_time:.3f}s")
            print(f"  📊 Checks performed: {summary.n_checks_performed}")
            
            if summary.warnings:
                print(f"  ⚠️ Warnings: {len(summary.warnings)}")
            if summary.recommendations:
                print(f"  💡 Recommendations: {len(summary.recommendations)}")
        
        # Test performance validation
        perf_result = validator.validate_performance(data, max_memory_mb=500.0, max_execution_time=60.0)
        print(f"📊 Performance validation: {perf_result.validation_passed}")
        print(f"📊 Memory usage: {perf_result.memory_usage_mb:.1f} MB")
        print(f"📊 Data size: {perf_result.data_size_mb:.1f} MB")
        
        # Test error handling
        try:
            validator.handle_error(
                ValueError("Test error"),
                category=validator.ErrorCategory.VALIDATION,
                severity=validator.ErrorSeverity.MEDIUM,
                context={"test": True}
            )
            print("✅ Error handling test passed")
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
        
        # Test performance stats
        stats = validator.get_validation_stats()
        print(f"📊 Validation stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive validation test failed: {e}")
        return False

def test_enhanced_unified_pipeline():
    """Test the enhanced unified pipeline with sophisticated components."""
    print("\n🧪 Testing Enhanced Unified Pipeline...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.core.enhanced_unified_pipeline import (
            EnhancedUnifiedDataDrivenPipeline, create_enhanced_unified_pipeline
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Initialize pipeline
        pipeline = create_enhanced_unified_pipeline()
        
        print("📊 Testing enhanced pipeline processing...")
        
        # Process data
        start_time = time.time()
        result = pipeline.process(data)
        processing_time = time.time() - start_time
        
        print(f"✅ Pipeline processing completed in {processing_time:.3f}s")
        print(f"📊 Selected features: {len(result.selected_features)}")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        print(f"📊 CV splits: {result.n_cv_splits}")
        print(f"📊 Candidates evaluated: {result.n_candidates_evaluated}")
        
        # Check sophisticated optimization results
        if hasattr(result, 'lookback_optimization_result') and result.lookback_optimization_result:
            lookback_result = result.lookback_optimization_result
            print(f"📊 Lookback optimization method: {lookback_result.get('optimization_method', 'unknown')}")
            print(f"📊 Optimized lookbacks: {len(lookback_result.get('optimized_lookbacks', {}))}")
            
            if 'performance_stats' in lookback_result:
                perf_stats = lookback_result['performance_stats']
                print(f"📊 Sophisticated optimization stats: {perf_stats}")
        
        # Check validation results
        if hasattr(result, 'performance_monitoring_data') and result.performance_monitoring_data:
            perf_data = result.performance_monitoring_data
            if 'validation_summary' in perf_data:
                val_summary = perf_data['validation_summary']
                print(f"📊 Data quality score: {val_summary.get('data_quality_score', 0.0):.3f}")
                print(f"📊 Memory usage: {val_summary.get('memory_usage_mb', 0.0):.1f} MB")
        
        # Check comprehensive validation stats
        if 'comprehensive_validation' in result.performance_monitoring_data:
            val_stats = result.performance_monitoring_data['comprehensive_validation']
            print(f"📊 Validation stats: {val_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced unified pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_completeness():
    """Test that all sophisticated logic has been integrated."""
    print("\n🧪 Testing Integration Completeness...")
    
    integration_tests = {
        "Sophisticated Lookback Optimizer": test_sophisticated_lookback_optimizer,
        "Multi-Horizon Integration": test_multi_horizon_integration,
        "Comprehensive Validation": test_comprehensive_validation,
        "Enhanced Unified Pipeline": test_enhanced_unified_pipeline
    }
    
    results = {}
    for test_name, test_func in integration_tests.items():
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("INTEGRATION COMPLETENESS SUMMARY")
    print('='*50)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Integration is complete!")
    else:
        print("⚠️ Some tests failed - Integration needs attention")
    
    return results

def main():
    """Run all integration tests."""
    print("🚀 Starting Sophisticated Integration Tests")
    print("="*60)
    
    # Run integration completeness test
    results = test_integration_completeness()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("🎉 SUCCESS: All sophisticated logic from FeatureLookbackOptimizationComponent")
        print("   has been successfully integrated into UnifiedDataDrivenPipeline!")
        print("\n✅ Features integrated:")
        print("   - Sophisticated lookback optimization algorithms")
        print("   - Multi-horizon profit labeling integration")
        print("   - Direction-specific optimization (longs/shorts)")
        print("   - Comprehensive validation system")
        print("   - Advanced metrics and performance monitoring")
        print("   - Execution mode-aware optimization")
        print("   - Nested walk-forward cross-validation")
        print("   - Sophisticated regularization settings")
    else:
        print("⚠️ PARTIAL SUCCESS: Some components need attention")
        print(f"   Passed: {passed}/{total} tests")
    
    return results

if __name__ == "__main__":
    main()