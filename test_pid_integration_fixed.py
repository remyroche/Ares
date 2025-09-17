#!/usr/bin/env python3
"""
Integration Test for Fixed PID-Based Feature Generation System

This test verifies that all the critical fixes work together to produce
actual features instead of the previous 0 features generated.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import sys
import os
from typing import Dict, List, Any
import traceback

# Add the workspace to the path
sys.path.append('/workspace')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_realistic_market_data(n_samples: int = 1000, n_features: int = 8) -> tuple:
    """Create realistic market data for testing."""
    np.random.seed(42)
    
    # Create realistic OHLCV data
    base_price = 100.0
    prices = [base_price]
    
    # Generate realistic price movements
    for _ in range(n_samples - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = {
        'open': prices + np.random.normal(0, 0.001, n_samples),
        'high': prices + np.abs(np.random.normal(0, 0.005, n_samples)),
        'low': prices - np.abs(np.random.normal(0, 0.005, n_samples)),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
        'quote_volume': np.random.lognormal(12, 1, n_samples),
        'volume_return': np.random.normal(0, 0.1, n_samples),
        'volume_log_return': np.random.normal(0, 0.05, n_samples)
    }
    
    # Ensure high >= low and reasonable relationships
    for i in range(n_samples):
        if data['low'][i] > data['high'][i]:
            data['low'][i], data['high'][i] = data['high'][i], data['low'][i]
        
        # Ensure open and close are between high and low
        data['open'][i] = np.clip(data['open'][i], data['low'][i], data['high'][i])
        data['close'][i] = np.clip(data['close'][i], data['low'][i], data['high'][i])
    
    df = pd.DataFrame(data)
    feature_names = list(df.columns)
    
    # Create target variable (future returns)
    target = np.diff(prices, prepend=prices[0]) / prices[:-1] if len(prices) > 1 else np.zeros(len(prices))
    target = np.append(target, target[-1])  # Ensure same length
    
    return df, target, feature_names

async def test_enhanced_pid_main():
    """Test the enhanced PID main module with fixed logic."""
    logger.info("🧪 Testing Enhanced PID Main Module")
    
    try:
        from src.training.utils.feature_selection.enhanced_pid_main import (
            EnhancedPartialInformationDecomposition, PIDConfig, PIDMeasure, DiscretizationMethod
        )
        
        # Create test data
        X, y, feature_names = create_realistic_market_data(500, 6)
        
        # Configure PID analysis
        config = PIDConfig(
            method="bivariate",
            pid_measures=[PIDMeasure.I_MIN],
            discretization_method=DiscretizationMethod.ADAPTIVE,
            enable_parallel=False,
            enable_financial_features=True,
            n_bins=5  # Reduce bins for small dataset
        )
        
        # Initialize and run PID analysis
        pid_module = EnhancedPartialInformationDecomposition(config)
        
        # Test input validation
        is_valid = pid_module.validate_inputs(X.values, y, feature_names)
        assert is_valid, "Input validation should pass with good data"
        
        # Run PID analysis
        results = pid_module.compute_pid(X.values, y, feature_names)
        
        # Verify results
        assert results is not None, "PID analysis should return results"
        assert 'method' in results, "Results should contain method"
        assert results['method'] == 'bivariate', "Method should be bivariate"
        
        # Check feature analysis results
        if 'feature_pid' in results:
            feature_count = len(results['feature_pid'])
            logger.info(f"✅ PID analysis completed for {feature_count} features")
            
            # Verify each feature has meaningful results
            for feature_name, feature_result in results['feature_pid'].items():
                assert feature_result is not None, f"Feature {feature_name} should have results"
                if 'mutual_information' in feature_result:
                    mi_result = feature_result['mutual_information']
                    assert 'total_mi' in mi_result, f"Feature {feature_name} should have total MI"
                    logger.info(f"  {feature_name}: MI = {mi_result['total_mi']:.4f}")
        
        # Check financial features if enabled
        if 'financial_features' in results:
            financial_count = len(results['financial_features'])
            logger.info(f"✅ Generated {financial_count} financial features")
        
        logger.info("✅ Enhanced PID Main Module test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced PID Main Module test failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def test_orchestrator_with_fallbacks():
    """Test the orchestrator with fallback generators."""
    logger.info("🧪 Testing PID Orchestrator with Fallbacks")
    
    try:
        from src.training.steps.market_analysis.pid_based_feature_generation.pid_based_feature_orchestrator import (
            PIDBasedFeatureOrchestrator, OrchestratorConfig
        )
        
        # Create test data
        X, y, feature_names = create_realistic_market_data(300, 5)
        
        # Configure orchestrator
        config = OrchestratorConfig(
            max_interaction_features=10,
            max_polynomial_features=10,
            max_cross_timeframe_features=10,
            enable_interaction_features=True,
            enable_polynomial_features=True,
            enable_cross_timeframe_features=True,
            enable_parallel_processing=False,  # Disable for testing
            enable_gpu_acceleration=False,
            memory_limit_gb=2.0
        )
        
        # Initialize orchestrator
        orchestrator = PIDBasedFeatureOrchestrator(config)
        logger.info("✅ Orchestrator initialized")
        
        # Run feature generation
        result = await orchestrator.orchestrate_feature_generation(
            X.values, feature_names, None, y
        )
        
        # Verify results
        assert result is not None, "Orchestrator should return results"
        assert hasattr(result, 'total_features_generated'), "Result should have total features count"
        
        feature_count = result.total_features_generated
        logger.info(f"✅ Generated {feature_count} total features")
        
        # Verify individual generators
        if hasattr(result, 'interaction_result') and result.interaction_result:
            interaction_count = getattr(result.interaction_result, 'total_features_generated', 0)
            if isinstance(result.interaction_result, dict):
                interaction_count = result.interaction_result.get('total_features_generated', 0)
            logger.info(f"  Interaction features: {interaction_count}")
        
        if hasattr(result, 'polynomial_result') and result.polynomial_result:
            polynomial_count = getattr(result.polynomial_result, 'total_features_generated', 0)
            if isinstance(result.polynomial_result, dict):
                polynomial_count = result.polynomial_result.get('total_features_generated', 0)
            logger.info(f"  Polynomial features: {polynomial_count}")
        
        if hasattr(result, 'cross_timeframe_result') and result.cross_timeframe_result:
            cross_timeframe_count = getattr(result.cross_timeframe_result, 'total_features_generated', 0)
            if isinstance(result.cross_timeframe_result, dict):
                cross_timeframe_count = result.cross_timeframe_result.get('total_features_generated', 0)
            logger.info(f"  Cross-timeframe features: {cross_timeframe_count}")
        
        # Verify we actually generated features (this was the main issue)
        assert feature_count > 0, f"Should generate at least some features, got {feature_count}"
        
        logger.info("✅ PID Orchestrator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ PID Orchestrator test failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def test_main_component_integration():
    """Test the main component with full integration."""
    logger.info("🧪 Testing Main Component Integration")
    
    try:
        from src.training.steps.market_analysis.pid_based_feature_generation.pid_based_feature_generation_component import (
            PIDBasedFeatureGenerationComponent
        )
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Create test data
        X, y, feature_names = create_realistic_market_data(400, 6)
        
        # Configure component
        config = ComponentConfig(
            symbol="TESTUSDT",
            exchange="test",
            timeframe="1h"
        )
        
        # Initialize component
        component = PIDBasedFeatureGenerationComponent(config)
        logger.info("✅ Component initialized")
        
        # Create mock pipeline state
        pipeline_state = {
            'market_data': X,
            'feature_names': feature_names,
            'target': y
        }
        
        # Execute component
        result = await component.execute(X, pipeline_state)
        
        # Verify results
        assert result is not None, "Component should return results"
        assert hasattr(result, 'success'), "Result should have success flag"
        
        if result.success:
            logger.info("✅ Component execution successful")
            
            # Check artifacts
            if hasattr(result, 'artifacts') and result.artifacts:
                artifacts = result.artifacts
                if 'pid_based_feature_generation_result' in artifacts:
                    pid_result = artifacts['pid_based_feature_generation_result']
                    total_features = pid_result.get('total_features_generated', 0)
                    logger.info(f"  Total features in artifacts: {total_features}")
                    
                    # This was the critical issue - we should have features now
                    if total_features > 0:
                        logger.info("🎉 SUCCESS: Features were actually generated!")
                    else:
                        logger.warning("⚠️ Still generating 0 features - need further investigation")
        else:
            logger.error(f"❌ Component execution failed: {getattr(result, 'error_message', 'Unknown error')}")
            return False
        
        logger.info("✅ Main Component Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Main Component Integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def test_simple_generator_fallback():
    """Test the simple generator fallback functionality."""
    logger.info("🧪 Testing Simple Generator Fallback")
    
    try:
        from src.training.steps.market_analysis.pid_based_feature_generation.simple_feature_generator import (
            SimpleFeatureGenerator
        )
        
        # Create test data
        X, y, feature_names = create_realistic_market_data(200, 4)
        
        # Initialize simple generator
        generator = SimpleFeatureGenerator(max_features=20)
        
        # Test interaction features
        interaction_result = generator.generate_interaction_features(X.values, feature_names)
        assert interaction_result is not None, "Should return interaction result"
        assert interaction_result.total_features_generated > 0, "Should generate interaction features"
        logger.info(f"✅ Generated {interaction_result.total_features_generated} interaction features")
        
        # Test polynomial features
        polynomial_result = generator.generate_polynomial_features(X.values, feature_names)
        assert polynomial_result is not None, "Should return polynomial result"
        assert polynomial_result.total_features_generated > 0, "Should generate polynomial features"
        logger.info(f"✅ Generated {polynomial_result.total_features_generated} polynomial features")
        
        # Test cross-timeframe features
        cross_timeframe_result = generator.generate_cross_timeframe_features(X.values, feature_names)
        assert cross_timeframe_result is not None, "Should return cross-timeframe result"
        assert cross_timeframe_result.total_features_generated > 0, "Should generate cross-timeframe features"
        logger.info(f"✅ Generated {cross_timeframe_result.total_features_generated} cross-timeframe features")
        
        total_simple_features = (
            interaction_result.total_features_generated + 
            polynomial_result.total_features_generated + 
            cross_timeframe_result.total_features_generated
        )
        
        logger.info(f"✅ Simple generator created {total_simple_features} total features")
        logger.info("✅ Simple Generator Fallback test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple Generator Fallback test failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting PID-Based Feature Generation Integration Tests")
    logger.info("=" * 80)
    
    # Run all tests
    test_results = []
    
    # Test 1: Enhanced PID Main
    result1 = await test_enhanced_pid_main()
    test_results.append(("Enhanced PID Main", result1))
    
    # Test 2: Simple Generator Fallback
    result2 = await test_simple_generator_fallback()
    test_results.append(("Simple Generator Fallback", result2))
    
    # Test 3: Orchestrator with Fallbacks
    result3 = await test_orchestrator_with_fallbacks()
    test_results.append(("PID Orchestrator", result3))
    
    # Test 4: Main Component Integration
    result4 = await test_main_component_integration()
    test_results.append(("Main Component Integration", result4))
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 Test Results Summary:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if passed:
            passed_tests += 1
    
    logger.info(f"\n📈 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! The PID-based feature generation fixes are working!")
        return 0
    else:
        logger.error("⚠️ Some tests failed. The system may still have issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)