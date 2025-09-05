#!/usr/bin/env python3
"""
Comprehensive Step06 Validation Test Script

This script demonstrates the enhanced step06 validation framework with:
- Function call validation and tracking
- Function-to-function call monitoring
- Comprehensive function completion reports
- Performance monitoring and analysis
- Error handling with detailed context
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'step06_validation_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_step06_validation_framework():
    """Test the step06 validation framework components."""
    logger.info("🚀 Starting Step06 Comprehensive Validation Test")
    
    try:
        # Import the validation orchestrator
        from training.steps.step06_validation_orchestrator import run_step06_comprehensive_validation
        
        # Test configuration
        config = {
            "step06_feature_engineering": {
                "use_matrix_optimizer": True,
                "force_regime_specific_periods": False,
                "momentum_volume_enabled": True,
                "trend_volatility_enabled": True,
                "oscillator_trend_enabled": True,
                "volume_price_enabled": True,
                "volatility_regime_enabled": True,
                "cross_timeframe_enabled": True,
                "regime_dependent_enabled": True,
                "max_interactions": 50,
                "min_importance": 0.01,
                "correlation_threshold": 0.8,
                "mutual_info_threshold": 0.05
            }
        }
        
        # Generate test data
        logger.info("📊 Generating test data...")
        test_data = generate_comprehensive_test_data()
        logger.info(f"✅ Generated test data: {test_data.shape}")
        
        # Run comprehensive validation
        logger.info("🔍 Running comprehensive step06 validation...")
        results = await run_step06_comprehensive_validation(
            config=config,
            test_data=test_data,
            output_dir="step06_validation_reports"
        )
        
        # Display results
        display_validation_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Step06 validation test failed: {e}")
        raise


def generate_comprehensive_test_data() -> pd.DataFrame:
    """Generate comprehensive test data for step06 validation."""
    logger.info("📊 Generating comprehensive test data...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic market data
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    
    # Generate price data with trends and volatility
    base_price = 100.0
    trend = np.linspace(0, 0.1, n_samples)  # 10% upward trend
    volatility = 0.02
    returns = np.random.normal(0, volatility, n_samples) + trend / n_samples
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "close": prices,
        "volume": np.random.uniform(1000, 50000, n_samples)
    }, index=dates)
    
    # Ensure OHLC consistency
    data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
    data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))
    
    # Add some additional features for testing
    data["returns"] = data["close"].pct_change()
    data["volatility"] = data["returns"].rolling(20).std()
    data["volume_ma"] = data["volume"].rolling(20).mean()
    data["volume_ratio"] = data["volume"] / data["volume_ma"]
    
    # Add regime labels for testing
    data["regime_label"] = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    
    logger.info(f"✅ Generated comprehensive test data: {data.shape}")
    logger.info(f"   Columns: {list(data.columns)}")
    logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")
    logger.info(f"   Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    
    return data


def display_validation_results(results: dict):
    """Display comprehensive validation results."""
    logger.info("📋 Step06 Validation Results Summary")
    logger.info("=" * 60)
    
    # Overall summary
    overall = results.get("overall_summary", {})
    logger.info(f"📊 Overall Summary:")
    logger.info(f"   Total Components: {overall.get('total_components', 0)}")
    logger.info(f"   Successful Components: {overall.get('successful_components', 0)}")
    logger.info(f"   Component Success Rate: {overall.get('component_success_rate', 0):.2%}")
    logger.info(f"   Total Tests: {overall.get('total_tests', 0)}")
    logger.info(f"   Successful Tests: {overall.get('successful_tests', 0)}")
    logger.info(f"   Test Success Rate: {overall.get('test_success_rate', 0):.2%}")
    
    # Component details
    component_validation = results.get("component_validation", {})
    logger.info(f"\n🔍 Component Details:")
    
    for component_name, component_result in component_validation.items():
        logger.info(f"\n   {component_name}:")
        logger.info(f"     Status: {component_result.get('status', 'unknown')}")
        
        validation_tests = component_result.get("validation_tests", {})
        logger.info(f"     Validation Tests: {len(validation_tests)}")
        
        for test_name, test_result in validation_tests.items():
            if isinstance(test_result, dict) and "status" in test_result:
                status_emoji = "✅" if test_result["status"] == "passed" else "❌"
                logger.info(f"       {status_emoji} {test_name}: {test_result['status']}")
                
                # Show additional details for passed tests
                if test_result["status"] == "passed":
                    for key, value in test_result.items():
                        if key != "status" and isinstance(value, (int, float, str)):
                            logger.info(f"         {key}: {value}")
        
        function_reports = component_result.get("function_reports", {})
        logger.info(f"     Function Reports: {len(function_reports)}")
        
        for report_name, report_data in function_reports.items():
            if isinstance(report_data, dict):
                logger.info(f"       📋 {report_name}: {len(report_data)} sections")
    
    # Test data info
    test_data_info = results.get("test_data_info", {})
    logger.info(f"\n📊 Test Data Information:")
    logger.info(f"   Shape: {test_data_info.get('shape', 'unknown')}")
    logger.info(f"   Columns: {len(test_data_info.get('columns', []))}")
    logger.info(f"   Data Types: {len(test_data_info.get('data_types', {}))}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Step06 Comprehensive Validation Test Completed")


async def test_individual_components():
    """Test individual step06 components separately."""
    logger.info("🔧 Testing individual step06 components...")
    
    try:
        # Test FeatureInteractionEngine
        logger.info("🔧 Testing FeatureInteractionEngine...")
        from training.steps.market_analysis.step06_feature_engineering import FeatureInteractionEngine
        
        config = {"step06_feature_engineering": {}}
        engine = FeatureInteractionEngine(config)
        
        test_data = generate_comprehensive_test_data()
        
        # Test technical indicators
        indicators = engine.extract_optimal_technical_indicators(test_data)
        logger.info(f"✅ Technical indicators extracted: {indicators.shape}")
        
        # Test correlation analysis
        correlation_results = engine.analyze_feature_correlations(indicators)
        logger.info(f"✅ Correlation analysis completed: {correlation_results.get('n_high_correlations', 0)} high correlations")
        
        # Test interaction features
        features_array = indicators.values
        feature_names = list(indicators.columns)
        interactions = engine.extract_interaction_features(features_array, feature_names, test_data)
        logger.info(f"✅ Interaction features extracted: {interactions.shape}")
        
        # Generate comprehensive report
        report = engine.generate_comprehensive_function_report()
        logger.info(f"✅ Comprehensive report generated: {len(report)} sections")
        
    except Exception as e:
        logger.error(f"❌ FeatureInteractionEngine test failed: {e}")
    
    try:
        # Test OptimizedTripleBarrierLabeling
        logger.info("🏷️ Testing OptimizedTripleBarrierLabeling...")
        from training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
        
        labeling = OptimizedTripleBarrierLabeling()
        test_data = generate_comprehensive_test_data()
        
        # Test vectorized labeling
        labeled_data = labeling.apply_triple_barrier_labeling_vectorized(test_data)
        logger.info(f"✅ Vectorized labeling completed: {labeled_data.shape}")
        logger.info(f"   Label distribution: {labeled_data['label'].value_counts().to_dict()}")
        
        # Test convenience method
        labels_only = labeling.apply_triple_barrier_labels(test_data)
        logger.info(f"✅ Convenience method completed: {len(labels_only)} labels")
        
        # Generate comprehensive report
        report = labeling.generate_comprehensive_labeling_report()
        logger.info(f"✅ Comprehensive labeling report generated: {len(report)} sections")
        
    except Exception as e:
        logger.error(f"❌ OptimizedTripleBarrierLabeling test failed: {e}")


async def main():
    """Main test function."""
    logger.info("🎯 Starting Step06 Comprehensive Validation Test Suite")
    logger.info(f"   Timestamp: {datetime.now()}")
    logger.info(f"   Python version: {sys.version}")
    logger.info(f"   Working directory: {os.getcwd()}")
    
    try:
        # Test individual components first
        await test_individual_components()
        
        # Test comprehensive validation framework
        results = await test_step06_validation_framework()
        
        logger.info("🎉 All Step06 validation tests completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"💥 Step06 validation test suite failed: {e}")
        raise


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())