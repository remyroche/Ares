#!/usr/bin/env python3
"""
Test Script for SR Feature Integration

This script tests the complete integration of SR feature extraction with the
parameter optimization engine and enhanced feature engineering pipeline.

Usage:
    python test_sr_feature_integration.py
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("🔍 Testing imports...")
    
    try:
        # Test SR feature extractor
        from src.feature_engineering.sr_feature_extractor import (
            SRFeatureExtractor, SRFeatureConfig, get_sr_feature_extractor, extract_sr_features
        )
        logger.info("✅ SR feature extractor imported successfully")
        
        # Test parameter optimization engine
        try:
            from src.utils.sr_clustering.parameter_optimization_engine import (
                ParameterOptimizationEngine, ParameterOptimizationConfig, 
                get_parameter_optimization_engine
            )
            logger.info("✅ Parameter optimization engine imported successfully")
            optimization_available = True
        except ImportError as e:
            logger.warning(f"⚠️ Parameter optimization engine not available: {e}")
            optimization_available = False
        
        # Test enhanced feature engineering
        from src.feature_engineering.step06_enhanced_feature_engineering_step import (
            EnhancedFeatureEngineeringStep
        )
        logger.info("✅ Enhanced feature engineering imported successfully")
        
        return True, optimization_available
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False, False

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test market data."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def create_test_sr_levels(data: pd.DataFrame) -> dict:
    """Create test SR levels."""
    window = 20
    highs = data['high'].rolling(window, center=True).max()
    lows = data['low'].rolling(window, center=True).min()
    
    swing_highs = data[data['high'] == highs]['high'].dropna().unique()
    swing_lows = data[data['low'] == lows]['low'].dropna().unique()
    
    support_levels = swing_lows[:5].tolist()
    resistance_levels = swing_highs[:5].tolist()
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'quality_scores': {
            f'level_{level:.6f}': np.random.uniform(0.3, 0.9) 
            for level in support_levels + resistance_levels
        }
    }

def test_sr_feature_extraction():
    """Test SR feature extraction."""
    logger.info("🔧 Testing SR feature extraction...")
    
    try:
        from src.feature_engineering.sr_feature_extractor import (
            SRFeatureExtractor, SRFeatureConfig, extract_sr_features
        )
        
        # Create test data
        data = create_test_data(500)
        sr_levels = create_test_sr_levels(data)
        
        # Test basic extraction
        start_time = time.time()
        sr_features = extract_sr_features(data, sr_levels)
        extraction_time = time.time() - start_time
        
        logger.info(f"✅ SR feature extraction completed in {extraction_time:.2f}s")
        logger.info(f"   Input data shape: {data.shape}")
        logger.info(f"   SR features shape: {sr_features.shape}")
        logger.info(f"   SR feature columns: {len(sr_features.columns)}")
        
        # Verify feature types
        expected_features = [
            'pivot_point', 'support_1', 'resistance_1', 'distance_to_support', 
            'distance_to_resistance', 'sr_strength_20'
        ]
        
        found_features = [col for col in expected_features if col in sr_features.columns]
        logger.info(f"   Found expected features: {found_features}")
        
        return True, sr_features
        
    except Exception as e:
        logger.error(f"❌ SR feature extraction failed: {e}")
        logger.error(traceback.format_exc())
        return False, None

def test_parameter_optimization(optimization_available: bool):
    """Test parameter optimization."""
    if not optimization_available:
        logger.info("⚠️ Skipping parameter optimization test (not available)")
        return True, None
    
    logger.info("⚙️ Testing parameter optimization...")
    
    try:
        from src.utils.sr_clustering.parameter_optimization_engine import (
            ParameterOptimizationEngine, ParameterOptimizationConfig, 
            get_parameter_optimization_engine
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Create optimization configuration
        opt_config = ParameterOptimizationConfig(
            optimization_method='adaptive_grid_search',
            n_trials=20,  # Small number for testing
            cv_folds=2,
            objective_metric='quality_score_correlation',
            enable_hardware_optimization=False,  # Disable for testing
            enable_parallel_processing=False
        )
        
        # Get optimization engine
        opt_engine = get_parameter_optimization_engine(opt_config)
        
        # Create mock backtest results
        mock_backtest_results = []
        for i in range(10):
            mock_result = type('MockResult', (), {
                'success_rate': np.random.uniform(0.3, 0.8),
                'avg_bounce_strength': np.random.uniform(0.001, 0.01),
                'total_volume_at_level': np.random.uniform(1000, 10000),
                'time_persistence': np.random.uniform(0.1, 0.9),
                'total_touches': np.random.randint(2, 10),
                'quality_score': np.random.uniform(0.2, 0.9)
            })()
            mock_backtest_results.append(mock_result)
        
        # Run optimization
        start_time = time.time()
        opt_result = opt_engine.optimize_parameters(mock_backtest_results, data)
        optimization_time = time.time() - start_time
        
        logger.info(f"✅ Parameter optimization completed in {optimization_time:.2f}s")
        logger.info(f"   Best score: {opt_result.best_score:.4f}")
        logger.info(f"   Best parameters: {list(opt_result.best_parameters.keys())}")
        
        return True, opt_result
        
    except Exception as e:
        logger.error(f"❌ Parameter optimization failed: {e}")
        logger.error(traceback.format_exc())
        return False, None

def test_feature_engineering_integration():
    """Test integration with enhanced feature engineering."""
    logger.info("🔗 Testing feature engineering integration...")
    
    try:
        from src.feature_engineering.step06_enhanced_feature_engineering_step import (
            EnhancedFeatureEngineeringStep
        )
        
        # Create test data
        data = create_test_data(1000)
        sr_levels = create_test_sr_levels(data)
        
        # Add regime labels
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        regime_labels = pd.cut(volatility, bins=3, labels=['low_vol', 'med_vol', 'high_vol'])
        data['regime_label'] = regime_labels
        
        # Create configuration
        config = {
            'step06_feature_engineering': {
                'use_technical_indicators': True,
                'use_interaction_features': True,
                'use_regime_features': True,
                'use_sr_features': True,
                'use_dynamic_lookback': True,
                'chunk_size': 1000,
                'max_features': 200,
                'polynomial_degree': 2,
                'correlation_threshold': 0.95,
                'memory_limit_mb': 500,
                'sr_detection_window': 20,
                'min_touches_required': 3,
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001,
                'volume_threshold_multiplier': 1.5,
                'use_pre_optimized_sr_parameters': True
            }
        }
        
        # Create feature engineering step
        feature_step = EnhancedFeatureEngineeringStep(config)
        
        # Simulate pipeline state
        feature_step.pipeline_state = {'sr_levels': sr_levels}
        
        # Process data
        start_time = time.time()
        processed_data = feature_step._process_data_split(data, 'train')
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Feature engineering integration completed in {processing_time:.2f}s")
        logger.info(f"   Original data shape: {data.shape}")
        logger.info(f"   Processed data shape: {processed_data.shape}")
        
        # Count SR features
        sr_feature_cols = [col for col in processed_data.columns 
                          if any(sr_term in col.lower() for sr_term in 
                                ['support', 'resistance', 'pivot', 'swing', 'sr_', 'bounce'])]
        logger.info(f"   SR features created: {len(sr_feature_cols)}")
        
        return True, processed_data
        
    except Exception as e:
        logger.error(f"❌ Feature engineering integration failed: {e}")
        logger.error(traceback.format_exc())
        return False, None

def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    logger.info("🔄 Testing end-to-end integration...")
    
    try:
        from src.feature_engineering.sr_feature_extractor import (
            SRFeatureExtractor, SRFeatureConfig, get_sr_feature_extractor
        )
        from src.feature_engineering.step06_enhanced_feature_engineering_step import (
            EnhancedFeatureEngineeringStep
        )
        
        # Create test data
        data = create_test_data(2000)
        sr_levels = create_test_sr_levels(data)
        
        # Add regime labels
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        regime_labels = pd.cut(volatility, bins=3, labels=['low_vol', 'med_vol', 'high_vol'])
        data['regime_label'] = regime_labels
        
        # Step 1: Extract SR features directly
        logger.info("   Step 1: Direct SR feature extraction...")
        sr_config = SRFeatureConfig(
            enable_basic_sr_features=True,
            enable_advanced_sr_features=True,
            enable_sr_bounce_signals=True,
            enable_sr_strength_calculation=True,
            enable_regime_aware_sr=True,
            use_pre_optimized_parameters=True
        )
        
        sr_extractor = get_sr_feature_extractor(sr_config)
        sr_features = sr_extractor.extract_sr_features(data, sr_levels, regime_labels)
        
        logger.info(f"   ✅ Direct SR extraction: {sr_features.shape[1]} features")
        
        # Step 2: Integration with feature engineering
        logger.info("   Step 2: Feature engineering integration...")
        config = {
            'step06_feature_engineering': {
                'use_technical_indicators': True,
                'use_interaction_features': True,
                'use_regime_features': True,
                'use_sr_features': True,
                'use_dynamic_lookback': True,
                'chunk_size': 2000,
                'max_features': 500,
                'polynomial_degree': 2,
                'correlation_threshold': 0.95,
                'memory_limit_mb': 1000,
                'sr_detection_window': 20,
                'min_touches_required': 3,
                'touch_tolerance': 0.002,
                'min_bounce_strength': 0.001,
                'volume_threshold_multiplier': 1.5,
                'use_pre_optimized_sr_parameters': True
            }
        }
        
        feature_step = EnhancedFeatureEngineeringStep(config)
        feature_step.pipeline_state = {'sr_levels': sr_levels}
        
        processed_data = feature_step._process_data_split(data, 'train')
        
        logger.info(f"   ✅ Feature engineering integration: {processed_data.shape[1]} total features")
        
        # Step 3: Verify SR features are present
        sr_feature_cols = [col for col in processed_data.columns 
                          if any(sr_term in col.lower() for sr_term in 
                                ['support', 'resistance', 'pivot', 'swing', 'sr_', 'bounce'])]
        
        logger.info(f"   ✅ SR features in final output: {len(sr_feature_cols)}")
        
        # Step 4: Verify data quality
        nan_count = processed_data.isna().sum().sum()
        inf_count = np.isinf(processed_data.select_dtypes(include=[np.number])).sum().sum()
        
        logger.info(f"   ✅ Data quality check: {nan_count} NaN values, {inf_count} infinite values")
        
        return True, processed_data
        
    except Exception as e:
        logger.error(f"❌ End-to-end integration failed: {e}")
        logger.error(traceback.format_exc())
        return False, None

def main():
    """Main test function."""
    logger.info("🚀 Starting SR Feature Integration Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: Imports
        logger.info("\n1. Testing Imports")
        logger.info("-" * 30)
        import_success, optimization_available = test_imports()
        test_results['imports'] = import_success
        
        if not import_success:
            logger.error("❌ Import test failed, stopping tests")
            return False
        
        # Test 2: SR Feature Extraction
        logger.info("\n2. Testing SR Feature Extraction")
        logger.info("-" * 30)
        sr_success, sr_features = test_sr_feature_extraction()
        test_results['sr_extraction'] = sr_success
        
        # Test 3: Parameter Optimization
        logger.info("\n3. Testing Parameter Optimization")
        logger.info("-" * 30)
        opt_success, opt_result = test_parameter_optimization(optimization_available)
        test_results['parameter_optimization'] = opt_success
        
        # Test 4: Feature Engineering Integration
        logger.info("\n4. Testing Feature Engineering Integration")
        logger.info("-" * 30)
        integration_success, processed_data = test_feature_engineering_integration()
        test_results['feature_engineering_integration'] = integration_success
        
        # Test 5: End-to-End Integration
        logger.info("\n5. Testing End-to-End Integration")
        logger.info("-" * 30)
        e2e_success, final_data = test_end_to_end_integration()
        test_results['end_to_end_integration'] = e2e_success
        
        # Summary
        logger.info("\n📊 Test Results Summary")
        logger.info("=" * 60)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"   {test_name}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! SR feature integration is working correctly.")
            return True
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed. Please check the errors above.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)