#!/usr/bin/env python3
"""
Production System Test

This script tests the complete production system to ensure everything is properly wired.
"""

import asyncio
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    logger.info("🔍 Testing imports...")
    
    try:
        # Test base validator imports
        from src.utils.base_validator import BaseValidator, DataValidator, ModelValidator, ConfigValidator
        logger.info("✅ Base validator imports successful")
        
        # Test early stopping imports
        from src.utils.standalone_early_stopping import (
            EarlyStoppingStrategy, AdaptivePatienceStrategy, ConvergenceBasedStrategy,
            PerformanceBasedStrategy, TimeBasedStrategy, TrialBasedStrategy, CompositeStrategy,
            create_default_strategy, EarlyStoppingConfig
        )
        logger.info("✅ Early stopping imports successful")
        
        # Test production factory imports
        from src.utils.production_factory import (
            ProductionMLFactory, ValidatorFactory, EarlyStoppingFactory,
            create_production_system, create_ml_validator_suite, create_early_stopping_suite
        )
        logger.info("✅ Production factory imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_validators():
    """Test validator functionality."""
    logger.info("🔍 Testing validators...")
    
    try:
        from src.utils.base_validator import DataValidator, ModelValidator, ConfigValidator
        
        # Test data validator
        data_validator = DataValidator('test_data', {
            'required_fields': ['features', 'targets'],
            'data_types': {'features': list, 'targets': list}
        })
        
        test_data = {'features': [[1, 2, 3], [4, 5, 6]], 'targets': [0, 1]}
        is_valid = data_validator.is_valid(test_data)
        assert is_valid == True, "Data validation should pass"
        logger.info("✅ Data validator test passed")
        
        # Test model validator
        model_validator = ModelValidator('test_model', {
            'required_methods': ['fit', 'predict']
        })
        
        class MockModel:
            def fit(self, X, y): pass
            def predict(self, X): return [0, 1]
        
        mock_model = MockModel()
        is_valid = model_validator.is_valid(mock_model)
        assert is_valid == True, "Model validation should pass"
        logger.info("✅ Model validator test passed")
        
        # Test config validator
        config_validator = ConfigValidator('test_config', {
            'required_keys': ['model_type'],
            'value_validators': {'learning_rate': lambda x: 0 < x < 1}
        })
        
        test_config = {'model_type': 'test', 'learning_rate': 0.01}
        is_valid = config_validator.is_valid(test_config)
        assert is_valid == True, "Config validation should pass"
        logger.info("✅ Config validator test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validator test failed: {e}")
        return False

def test_early_stopping():
    """Test early stopping strategies."""
    logger.info("🔍 Testing early stopping strategies...")
    
    try:
        from src.utils.standalone_early_stopping import (
            AdaptivePatienceStrategy, ConvergenceBasedStrategy, PerformanceBasedStrategy,
            TimeBasedStrategy, TrialBasedStrategy, CompositeStrategy, EarlyStoppingConfig
        )
        
        # Test adaptive patience strategy
        adaptive = AdaptivePatienceStrategy()
        history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.845, 0.847]
        should_stop = adaptive.should_stop(history, 10)
        logger.info(f"✅ Adaptive patience strategy test: should_stop = {should_stop}")
        
        # Test convergence strategy
        convergence = ConvergenceBasedStrategy()
        should_stop = convergence.should_stop(history, 10)
        logger.info(f"✅ Convergence strategy test: should_stop = {should_stop}")
        
        # Test performance strategy
        performance = PerformanceBasedStrategy()
        should_stop = performance.should_stop(history, 10)
        logger.info(f"✅ Performance strategy test: should_stop = {should_stop}")
        
        # Test time strategy
        time_strategy = TimeBasedStrategy()
        should_stop = time_strategy.should_stop(history, 10)
        logger.info(f"✅ Time strategy test: should_stop = {should_stop}")
        
        # Test trial strategy
        trial = TrialBasedStrategy()
        should_stop = trial.should_stop(history, 10)
        logger.info(f"✅ Trial strategy test: should_stop = {should_stop}")
        
        # Test composite strategy
        composite = CompositeStrategy([adaptive, convergence, performance])
        should_stop = composite.should_stop(history, 10)
        logger.info(f"✅ Composite strategy test: should_stop = {should_stop}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Early stopping test failed: {e}")
        return False

def test_production_factory():
    """Test production factory functionality."""
    logger.info("🔍 Testing production factory...")
    
    try:
        from src.utils.production_factory import create_production_system
        
        # Create production system
        system = create_production_system()
        logger.info("✅ Production system created")
        
        # Test validators
        data_validator = system.get_validator('data')
        assert data_validator is not None, "Data validator should be available"
        logger.info("✅ Data validator retrieved")
        
        model_validator = system.get_validator('model')
        assert model_validator is not None, "Model validator should be available"
        logger.info("✅ Model validator retrieved")
        
        config_validator = system.get_validator('config')
        assert config_validator is not None, "Config validator should be available"
        logger.info("✅ Config validator retrieved")
        
        # Test early stopping strategies
        early_stopping = system.get_default_early_stopping_strategy()
        assert early_stopping is not None, "Default early stopping strategy should be available"
        logger.info("✅ Default early stopping strategy retrieved")
        
        # Test system summary
        summary = system.get_system_summary()
        assert 'total_components' in summary, "System summary should include total components"
        assert summary['total_components'] > 0, "System should have components"
        logger.info(f"✅ System summary: {summary['total_components']} components")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Production factory test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    logger.info("🔍 Testing async functionality...")
    
    try:
        from src.utils.base_validator import DataValidator
        
        # Create validator
        validator = DataValidator('async_test', {
            'required_fields': ['data'],
            'data_types': {'data': list}
        })
        
        # Test async validation
        test_data = {'data': [1, 2, 3, 4, 5]}
        result = await validator.validate(test_data)
        assert result['success'] == True, "Async validation should succeed"
        logger.info("✅ Async validation test passed")
        
        # Test validation history
        history = validator.get_validation_history()
        assert len(history) > 0, "Validation history should not be empty"
        logger.info(f"✅ Validation history test passed: {len(history)} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Async functionality test failed: {e}")
        return False

def test_integration():
    """Test complete integration."""
    logger.info("🔍 Testing complete integration...")
    
    try:
        from src.utils.production_factory import create_production_system
        
        # Create system
        system = create_production_system()
        
        # Test data validation
        data_validator = system.get_validator('data')
        test_data = {'features': [[1, 2, 3], [4, 5, 6]], 'targets': [0, 1]}
        data_valid = data_validator.is_valid(test_data)
        logger.info(f"✅ Data validation: {'PASSED' if data_valid else 'FAILED'}")
        
        # Test early stopping
        early_stopping = system.get_default_early_stopping_strategy()
        history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.845, 0.847]
        should_stop = early_stopping.should_stop(history, 10)
        logger.info(f"✅ Early stopping: {'STOP' if should_stop else 'CONTINUE'}")
        
        # Test system monitoring
        summary = system.get_system_summary()
        logger.info(f"✅ System monitoring: {summary['total_components']} components active")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting production system tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Validator Test", test_validators),
        ("Early Stopping Test", test_early_stopping),
        ("Production Factory Test", test_production_factory),
        ("Integration Test", test_integration),
    ]
    
    results = {}
    
    # Run synchronous tests
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Run async test
    logger.info(f"\n{'='*50}")
    logger.info("Running Async Functionality Test")
    logger.info(f"{'='*50}")
    
    try:
        result = asyncio.run(test_async_functionality())
        results["Async Test"] = result
        if result:
            logger.info("✅ Async Functionality Test PASSED")
        else:
            logger.error("❌ Async Functionality Test FAILED")
    except Exception as e:
        logger.error(f"❌ Async Functionality Test FAILED with exception: {e}")
        results["Async Test"] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Production system is fully wired and ready!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)