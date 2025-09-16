#!/usr/bin/env python3
"""
Test script for the new coarse/fine grid + Optuna TPE optimization implementations.

This script tests all three implementations:
1. Final Parameters Optimization
2. HPO Utils
3. Hierarchical HPO

Usage:
    python test_coarse_fine_optimization.py
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test datasets for optimization."""
    logger.info("📊 Creating test datasets...")
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Classification dataset
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=42
    )
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    logger.info(f"✅ Created datasets - Regression: {X_reg_train.shape}, Classification: {X_clf_train.shape}")
    
    return {
        'regression': {
            'X_train': X_reg_train, 'X_test': X_reg_test,
            'y_train': y_reg_train, 'y_test': y_reg_test
        },
        'classification': {
            'X_train': X_clf_train, 'X_test': X_clf_test,
            'y_train': y_clf_train, 'y_test': y_clf_test
        }
    }

def test_final_parameters_optimization():
    """Test the Final Parameters Optimization with coarse/fine grid + Optuna TPE."""
    logger.info("🧪 Testing Final Parameters Optimization...")
    
    try:
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        from src.utils.nonlinear_optimization_helpers import NonLinearConfig
        
        # Create test configuration
        config = {
            'n_trials': 20,  # Small number for testing
            'timeout': 60,   # 1 minute timeout
            'study_name': 'test_final_params',
            'use_nonlinear_optimization': True
        }
        
        # Create non-linear config
        nonlinear_config = NonLinearConfig(
            use_log_sampling=True,
            use_fractional_powers=True,
            use_sigmoid_transforms=True,
            use_adaptive_transforms=True
        )
        
        # Initialize optimizer
        optimizer = FinalParametersOptimizer(config, nonlinear_config)
        
        # Create mock calibration results
        calibration_results = {
            'confidence_scores': np.random.random(100),
            'position_sizes': np.random.random(100),
            'leverage_factors': np.random.random(100)
        }
        
        # Test optimization for a single category
        logger.info("🎯 Testing confidence parameter optimization...")
        start_time = time.time()
        
        result = await optimizer._optimize_category('confidence', calibration_results)
        
        optimization_time = time.time() - start_time
        
        if result and 'best_params' in result:
            logger.info(f"✅ Final Parameters Optimization test passed!")
            logger.info(f"   📈 Best score: {result.get('best_value', 0):.4f}")
            logger.info(f"   ⏱️ Optimization time: {optimization_time:.2f}s")
            logger.info(f"   🎯 Best stage: {result.get('best_stage', 'unknown')}")
            logger.info(f"   📊 Method: {result.get('optimization_method', 'unknown')}")
            
            # Log stage results
            if 'coarse_result' in result:
                logger.info(f"   🔍 Coarse grid score: {result['coarse_result'].get('best_score', 0):.4f}")
            if 'fine_result' in result:
                logger.info(f"   🔍 Fine grid score: {result['fine_result'].get('best_score', 0):.4f}")
            if 'optuna_result' in result:
                logger.info(f"   🔍 Optuna TPE score: {result['optuna_result'].get('best_score', 0):.4f}")
            
            return True
        else:
            logger.error("❌ Final Parameters Optimization test failed - No results returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Final Parameters Optimization test failed: {e}")
        return False

def test_hpo_utils():
    """Test the HPO Utils with coarse/fine grid + Optuna TPE."""
    logger.info("🧪 Testing HPO Utils...")
    
    try:
        from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
        from src.utils.nonlinear_optimization_helpers import NonLinearConfig
        
        # Create test data
        test_data = create_test_data()
        X_train = test_data['regression']['X_train']
        y_train = test_data['regression']['y_train']
        
        # Create configuration
        config = {
            'enable_parallel': False,  # Disable parallel for testing
            'max_workers': 1,
            'enable_monitoring': True,
            'use_nonlinear_optimization': True
        }
        
        # Create non-linear config
        nonlinear_config = NonLinearConfig(
            use_log_sampling=True,
            use_fractional_powers=True,
            use_sigmoid_transforms=False,  # Disable for simplicity
            use_adaptive_transforms=True
        )
        
        # Initialize HPO
        hpo = HyperparameterOptimization(config, nonlinear_config)
        
        # Define model factory
        def model_factory(**params):
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42,
                n_jobs=1
            )
        
        # Define search space
        search_space = {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
            'max_depth': {'type': 'int', 'low': 5, 'high': 15},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
        }
        
        logger.info("🎯 Testing staged HPO with coarse/fine/optuna...")
        start_time = time.time()
        
        result = hpo.staged_hpo(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            search_space=search_space,
            coarse_strategy='grid',
            coarse_grid_points=3,
            fine_grid_points=5,
            bayes_n_trials=15,  # Small number for testing
            scoring='neg_mean_squared_error',
            subsample_rate=0.5,  # Use subset for faster testing
            finalize_refine=True
        )
        
        optimization_time = time.time() - start_time
        
        if result and 'final_params' in result and 'error' not in result:
            logger.info(f"✅ HPO Utils test passed!")
            logger.info(f"   📈 Final score: {result.get('final_score', 0):.4f}")
            logger.info(f"   ⏱️ Total optimization time: {optimization_time:.2f}s")
            logger.info(f"   🎯 Best stage: {result.get('best_stage', 'unknown')}")
            logger.info(f"   📊 Method: {result.get('optimization_method', 'unknown')}")
            
            # Log stage results
            if 'coarse_results' in result:
                logger.info(f"   🔍 Coarse grid score: {result['coarse_results'].get('best_score', 0):.4f}")
            if 'fine_results' in result:
                logger.info(f"   🔍 Fine grid score: {result['fine_results'].get('best_score', 0):.4f}")
            if 'optuna_results' in result:
                logger.info(f"   🔍 Optuna TPE score: {result['optuna_results'].get('best_score', 0):.4f}")
            
            # Log timing breakdown
            logger.info(f"   ⏱️ Coarse time: {result.get('coarse_time', 0):.2f}s")
            logger.info(f"   ⏱️ Fine time: {result.get('fine_time', 0):.2f}s")
            logger.info(f"   ⏱️ Optuna time: {result.get('optuna_time', 0):.2f}s")
            
            return True
        else:
            logger.error(f"❌ HPO Utils test failed - Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ HPO Utils test failed: {e}")
        return False

def test_hierarchical_hpo():
    """Test the Hierarchical HPO with coarse/fine grid + Optuna TPE."""
    logger.info("🧪 Testing Hierarchical HPO...")
    
    try:
        from src.utils.ml_common.optimization.hierarchical_hpo import (
            HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
        )
        
        # Create test data
        test_data = create_test_data()
        X_train = test_data['regression']['X_train']
        y_train = test_data['regression']['y_train']
        X_val = test_data['regression']['X_test']
        y_val = test_data['regression']['y_test']
        
        # Define base models
        base_models = {
            'rf1': RandomForestRegressor(random_state=42, n_jobs=1),
            'rf2': RandomForestRegressor(random_state=43, n_jobs=1)
        }
        
        # Define meta models
        meta_models = {
            'meta_rf': RandomForestRegressor(random_state=44, n_jobs=1)
        }
        
        # Define search spaces
        base_search_spaces = {
            'rf1': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 150},
                'max_depth': {'type': 'int', 'low': 5, 'high': 10}
            },
            'rf2': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 150},
                'max_depth': {'type': 'int', 'low': 5, 'high': 10}
            }
        }
        
        meta_search_spaces = {
            'meta_rf': {
                'n_estimators': {'type': 'int', 'low': 30, 'high': 100},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8}
            }
        }
        
        # Create phase configurations
        phase1_config = HPOPhaseConfig(
            phase_name="base_models",
            models=base_models,
            search_spaces=base_search_spaces,
            n_trials=10,  # Small number for testing
            timeout_seconds=60,
            enable_pruning=True,
            cv_folds=3,
            scoring_metric='neg_mean_squared_error'
        )
        
        phase2_config = HPOPhaseConfig(
            phase_name="meta_models",
            models=meta_models,
            search_spaces=meta_search_spaces,
            n_trials=8,  # Small number for testing
            timeout_seconds=60,
            enable_pruning=True,
            cv_folds=3,
            scoring_metric='neg_mean_squared_error'
        )
        
        # Create hierarchical HPO configuration
        hpo_config = HierarchicalHPOConfig(
            phase1_config=phase1_config,
            phase2_config=phase2_config,
            enable_caching=False,  # Disable caching for testing
            enable_parallel=False,  # Disable parallel for testing
            max_workers=1,
            random_state=42
        )
        
        # Initialize hierarchical HPO
        hierarchical_hpo = HierarchicalHPO(hpo_config)
        
        logger.info("🎯 Testing hierarchical HPO with coarse/fine/optuna...")
        start_time = time.time()
        
        result = hierarchical_hpo.optimize_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        optimization_time = time.time() - start_time
        
        if result and 'base_models' in result and 'meta_models' in result:
            logger.info(f"✅ Hierarchical HPO test passed!")
            logger.info(f"   ⏱️ Total optimization time: {optimization_time:.2f}s")
            logger.info(f"   📊 Phase 1 time: {result.get('phase1_time', 0):.2f}s")
            logger.info(f"   📊 Phase 2 time: {result.get('phase2_time', 0):.2f}s")
            logger.info(f"   🔧 Base models optimized: {len(result['base_models'])}")
            logger.info(f"   🔧 Meta models optimized: {len(result['meta_models'])}")
            
            # Log optimization history details
            if hasattr(hierarchical_hpo, 'phase1_result') and hierarchical_hpo.phase1_result:
                for history in hierarchical_hpo.phase1_result.optimization_history:
                    logger.info(f"   📈 {history['model_name']}: {history['best_score']:.4f} (stage: {history.get('best_stage', 'unknown')})")
            
            return True
        else:
            logger.error("❌ Hierarchical HPO test failed - No results returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hierarchical HPO test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting coarse/fine grid + Optuna TPE optimization tests...")
    
    test_results = {}
    
    # Test 1: Final Parameters Optimization
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Final Parameters Optimization")
    logger.info("="*60)
    test_results['final_params'] = await test_final_parameters_optimization()
    
    # Test 2: HPO Utils
    logger.info("\n" + "="*60)
    logger.info("TEST 2: HPO Utils")
    logger.info("="*60)
    test_results['hpo_utils'] = test_hpo_utils()
    
    # Test 3: Hierarchical HPO
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Hierarchical HPO")
    logger.info("="*60)
    test_results['hierarchical_hpo'] = test_hierarchical_hpo()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Coarse/fine grid + Optuna TPE implementations are working correctly.")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed. Please check the implementations.")
    
    return test_results

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())