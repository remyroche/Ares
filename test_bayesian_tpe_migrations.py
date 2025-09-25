#!/usr/bin/env python3
"""
Test script for Bayesian TPE optimizer migrations.

This script tests the migrated optimizers to ensure they work correctly
with the new Bayesian TPE optimizer implementation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bayesian_tpe_optimizer():
    """Test the core Bayesian TPE optimizer."""
    try:
        from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
            BayesianTPEOptimizer,
            BayesianTPEConfig
        )
        
        logger.info("🧪 Testing Bayesian TPE Optimizer...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Define simple search space
        search_space = {
            'param1': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'param2': {'type': 'int', 'low': 1, 'high': 10},
            'param3': {'type': 'categorical', 'choices': ['option1', 'option2']}
        }
        
        # Define simple objective function
        def objective_function(params: Dict[str, Any], **kwargs) -> float:
            # Simple objective: maximize param1 + param2/10 + (1 if param3 == 'option1' else 0)
            score = params['param1'] + params['param2'] / 10.0
            if params['param3'] == 'option1':
                score += 1.0
            return score
        
        # Configure optimizer
        config = BayesianTPEConfig(
            n_trials=5,  # Small number for testing
            timeout_seconds=30,
            enable_grid_search=True,
            coarse_grid_points=2,
            fine_grid_points=2,
            backend='optuna',
            enable_parallel=False,  # Disable parallel for testing
            log_level='INFO'
        )
        
        # Run optimization
        optimizer = BayesianTPEOptimizer(config)
        result = optimizer.optimize(objective_function, search_space)
        
        if result.success:
            logger.info(f"✅ Bayesian TPE Optimizer test passed!")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Best params: {result.best_params}")
            logger.info(f"   Optimization time: {result.optimization_time:.2f}s")
            return True
        else:
            logger.error(f"❌ Bayesian TPE Optimizer test failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Bayesian TPE Optimizer test failed with exception: {e}")
        return False

def test_hpo_utils_migration():
    """Test the migrated HPO utils."""
    try:
        from src.utils.nas_tas.advanced_hpo_utils import HyperparameterOptimization
        
        logger.info("🧪 Testing HPO Utils migration...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        # Create simple model factory
        def model_factory(**params):
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        
        # Define search space
        search_space = {
            'fit_intercept': {'type': 'categorical', 'choices': [True, False]},
            'normalize': {'type': 'categorical', 'choices': [True, False]}
        }
        
        # Configure HPO
        config = {
            'enable_parallel': False,
            'max_workers': 1,
            'use_nonlinear_optimization': False
        }
        
        hpo = HyperparameterOptimization(config=config)
        
        # Test Bayesian optimization
        result = hpo.bayesian_optimization(
            model_factory=model_factory,
            X=X, y=y,
            search_space=search_space,
            n_trials=3,  # Small number for testing
            timeout=30
        )
        
        if 'error' not in result:
            logger.info(f"✅ HPO Utils migration test passed!")
            logger.info(f"   Best score: {result.get('best_score', 'N/A')}")
            logger.info(f"   Best params: {result.get('best_params', 'N/A')}")
            return True
        else:
            logger.error(f"❌ HPO Utils migration test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ HPO Utils migration test failed with exception: {e}")
        return False

async def test_final_parameters_optimization_migration():
    """Test the migrated Final Parameters Optimization."""
    try:
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        
        logger.info("🧪 Testing Final Parameters Optimization migration...")
        
        # Create simple test data
        calibration_results = {
            'model_performance': {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75},
            'confidence_scores': np.random.rand(100),
            'predictions': np.random.rand(100)
        }
        
        # Configure optimizer
        config = {
            'n_trials': 3,  # Small number for testing
            'timeout': 30,
            'use_nonlinear_optimization': False
        }
        
        optimizer = FinalParametersOptimizer(config)
        
        # Test optimization for a simple category
        result = await optimizer._optimize_category(
            category='confidence',
            calibration_results=calibration_results
        )
        
        if result and 'best_params' in result:
            logger.info(f"✅ Final Parameters Optimization migration test passed!")
            logger.info(f"   Best score: {result.get('best_value', 'N/A')}")
            logger.info(f"   Best params: {result.get('best_params', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Final Parameters Optimization migration test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Final Parameters Optimization migration test failed with exception: {e}")
        return False

def test_attention_network_optimizer_migration():
    """Test the migrated Attention Network Optimizer."""
    try:
        from src.training.steps.model_training.bayesian_optimization_msm import AttentionNetworkOptimizer
        
        logger.info("🧪 Testing Attention Network Optimizer migration...")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        # Create simple base model
        from sklearn.linear_model import LinearRegression
        base_model = LinearRegression()
        
        # Configure optimizer
        config = {
            'n_trials': 3,  # Small number for testing
            'timeout': 30
        }
        
        optimizer = AttentionNetworkOptimizer(config)
        
        # Test optimization
        result = optimizer.optimize(X, y, base_model)
        
        if result.get('success', False):
            logger.info(f"✅ Attention Network Optimizer migration test passed!")
            logger.info(f"   Best score: {result.get('best_score', 'N/A')}")
            logger.info(f"   Best params: {result.get('best_params', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Attention Network Optimizer migration test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Attention Network Optimizer migration test failed with exception: {e}")
        return False

async def main():
    """Run all migration tests."""
    logger.info("🚀 Starting Bayesian TPE Migration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Bayesian TPE Optimizer", test_bayesian_tpe_optimizer),
        ("HPO Utils Migration", test_hpo_utils_migration),
        ("Final Parameters Optimization Migration", test_final_parameters_optimization_migration),
        ("Attention Network Optimizer Migration", test_attention_network_optimizer_migration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            if test_name == "Final Parameters Optimization Migration":
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All migration tests passed! The Bayesian TPE optimizer is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())