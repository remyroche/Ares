#!/usr/bin/env python3
"""
Fully Integrated NAS Regime Example

This example demonstrates how all the enhanced utility tools are fully wired
and used together in the NAS regime detection system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data(n_points=1000):
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='H')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_points)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV with realistic relationships
    data = pd.DataFrame(index=dates)
    data['open'] = prices
    data['close'] = prices * (1 + np.random.normal(0, 0.005, n_points))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    data['volume'] = np.random.uniform(1000, 10000, n_points)
    
    # Ensure high >= low and high >= open/close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def demonstrate_full_integration():
    """Demonstrate full integration of all enhanced utility tools."""
    logger.info("🚀 Starting Fully Integrated NAS Regime Example...")
    
    try:
        # Import the enhanced detector
        from core.perfect_nas_regime_detector import PerfectNASRegimeDetector
        from core.perfect_nas_config import EnhancedPerfectNASConfig
        
        # Create sample market data
        logger.info("📊 Creating sample market data...")
        market_data = create_sample_market_data(500)
        logger.info(f"✅ Created market data with {len(market_data)} points")
        logger.info(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
        logger.info(f"   Price range: ${market_data['low'].min():.2f} - ${market_data['high'].max():.2f}")
        
        # Initialize enhanced configuration
        config = EnhancedPerfectNASConfig(
            primary_architecture="NEURAL_ODE",
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_meta_learning=True,
            search_strategy="ENHANCED_BAYESIAN"
        )
        
        # Initialize the detector with full utility integration
        logger.info("🔧 Initializing Perfect NAS Regime Detector with enhanced utilities...")
        detector = PerfectNASRegimeDetector(config)
        
        # Check utility integration status
        metrics = detector.get_performance_metrics()
        logger.info("📈 Enhanced Utilities Status:")
        for utility, status in metrics['enhanced_utilities_status'].items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {utility}: {'Available' if status else 'Not available'}")
        
        # Test data quality reporting
        logger.info("🔍 Testing data quality reporting...")
        quality_report = detector.get_data_quality_report(market_data)
        logger.info(f"   Data quality score: {quality_report.get('data_quality_score', 'N/A')}")
        logger.info(f"   Missing values: {quality_report.get('missing_values', 'N/A')}")
        logger.info(f"   Warnings: {len(quality_report.get('warnings', []))}")
        
        # Test enhanced features calculation
        logger.info("🧮 Testing enhanced features calculation...")
        market_data_array = market_data.values
        enhanced_features = detector.get_enhanced_features(market_data_array, window=20)
        logger.info(f"   Enhanced features calculated: {len(enhanced_features)} types")
        for feature_name, feature_data in enhanced_features.items():
            logger.info(f"   - {feature_name}: shape {feature_data.shape}")
        
        # Perform regime detection with full integration
        logger.info("🎯 Performing regime detection with full utility integration...")
        
        # Generate timestamps
        timestamps = np.array([(dt - market_data.index[0]).total_seconds() / 3600 for dt in market_data.index])
        
        # Detect regimes
        result = detector.detect_regimes(
            market_data=market_data,
            timestamps=timestamps,
            optimize_architecture=True,
            enable_meta_learning=True,
            learn_thresholds=True
        )
        
        # Analyze results
        logger.info("📊 Regime Detection Results:")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Execution time: {result.execution_time:.2f} seconds")
        
        if result.success:
            logger.info(f"   Regime predictions: {len(result.regime_predictions)} points")
            logger.info(f"   Unique regimes detected: {len(np.unique(result.regime_predictions))}")
            logger.info(f"   Regime probabilities shape: {result.regime_probabilities.shape}")
            logger.info(f"   Economic significance scores: {len(result.economic_significance_scores)} points")
            logger.info(f"   Trading viability scores: {len(result.trading_viability_scores)} points")
            logger.info(f"   Regime stability scores: {len(result.regime_stability_scores)} points")
            
            # Show regime distribution
            regime_counts = np.bincount(result.regime_predictions)
            logger.info("   Regime distribution:")
            for regime_id, count in enumerate(regime_counts):
                percentage = (count / len(result.regime_predictions)) * 100
                logger.info(f"     Regime {regime_id}: {count} points ({percentage:.1f}%)")
        
        # Test state persistence
        logger.info("💾 Testing state persistence...")
        state_saved = detector.save_detector_state("detector_state_test.json")
        logger.info(f"   State saved: {state_saved}")
        
        if state_saved:
            # Create a new detector and load state
            new_detector = PerfectNASRegimeDetector(config)
            state_loaded = new_detector.load_detector_state("detector_state_test.json")
            logger.info(f"   State loaded: {state_loaded}")
        
        # Test data saving (if data operations available)
        if detector.data_operations:
            logger.info("💾 Testing processed data saving...")
            data_saved = detector.save_processed_data(market_data, "SAMPLE", "1h")
            logger.info(f"   Processed data saved: {data_saved}")
        
        # Show enhanced utilities metadata
        if result.metadata and 'enhanced_utilities' in result.metadata:
            logger.info("🔧 Enhanced Utilities Used in Detection:")
            for utility, status in result.metadata['enhanced_utilities'].items():
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {utility}: {'Used' if status else 'Not used'}")
        
        # Clean up test files
        test_files = ["detector_state_test.json"]
        for filename in test_files:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"🗑️ Cleaned up test file: {filename}")
        
        logger.info("🎉 Fully integrated example completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fully integrated example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_individual_utilities():
    """Demonstrate individual utility components."""
    logger.info("🔧 Demonstrating Individual Utility Components...")
    
    try:
        # Test Enhanced Matrix Operations
        logger.info("🧮 Testing Enhanced Matrix Operations...")
        from core.enhanced_matrix_operations import EnhancedMatrixOperations
        
        matrix_ops = EnhancedMatrixOperations(enable_gpu=False, enable_m1_optimization=False)
        test_data = np.random.randn(100, 5)
        
        # Test normalization
        normalized = matrix_ops.normalize_data(test_data, method='robust')
        logger.info(f"   ✅ Data normalization: {normalized.shape}")
        
        # Test correlation matrix
        corr_matrix = matrix_ops.calculate_correlation_matrix(test_data)
        logger.info(f"   ✅ Correlation matrix: {corr_matrix.shape}")
        
        # Test enhanced features
        features = matrix_ops.calculate_enhanced_features(test_data, window=10)
        logger.info(f"   ✅ Enhanced features: {len(features)} types")
        
        # Test Enhanced Data Operations
        logger.info("📊 Testing Enhanced Data Operations...")
        from core.enhanced_data_operations import EnhancedDataOperations
        
        data_ops = EnhancedDataOperations(enable_validation=True)
        sample_data = create_sample_market_data(100)
        
        # Test validation
        validation = data_ops.validate_market_data(sample_data)
        logger.info(f"   ✅ Data validation: {validation['is_valid']}")
        
        # Test processing
        processed = data_ops.process_market_data(sample_data)
        logger.info(f"   ✅ Data processing: {processed.shape}")
        
        # Test Enhanced ML Common Integration
        logger.info("🤖 Testing Enhanced ML Common Integration...")
        from core.enhanced_ml_common_integration import EnhancedMLCommonIntegration, MLCommonConfig
        
        ml_config = MLCommonConfig(enable_hardware_optimization=False, enable_m1_optimization=False)
        ml_integration = EnhancedMLCommonIntegration(ml_config)
        
        # Test validation
        ml_validation = ml_integration.validate_data(test_data, 'market_data')
        logger.info(f"   ✅ ML validation: {ml_validation['is_valid']}")
        
        # Test feature selection
        feature_result = ml_integration.select_features(test_data)
        logger.info(f"   ✅ Feature selection: {len(feature_result.get('selected_features', []))} features")
        
        logger.info("✅ Individual utility components test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Individual utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all demonstrations."""
    logger.info("🚀 Starting NAS Regime Enhanced Utilities Demonstration...")
    
    # Test individual utilities first
    individual_success = demonstrate_individual_utilities()
    
    # Then test full integration
    integration_success = demonstrate_full_integration()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEMONSTRATION SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"Individual Utilities: {'✅ PASSED' if individual_success else '❌ FAILED'}")
    logger.info(f"Full Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if individual_success and integration_success:
        logger.info("🎉 All demonstrations passed! Enhanced utilities are fully integrated.")
        return 0
    else:
        logger.error("⚠️ Some demonstrations failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())