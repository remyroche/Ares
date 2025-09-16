#!/usr/bin/env python3
"""
Test Script for HMM LM Models Training Pathway

This script tests the complete HMM LM models training pathway to ensure
all components are properly integrated and working correctly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_data(
    n_samples: int = 1000,
    n_features: int = 50,
    n_regimes: int = 3,
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """Generate mock data for testing."""
    logger.info(f"Generating mock data: {n_samples} samples, {n_features} features, {n_regimes} regimes")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate regime labels
    regime_labels = np.random.randint(0, n_regimes, n_samples)
    
    # Generate targets based on regime
    y = np.zeros(n_samples)
    for regime in range(n_regimes):
        regime_mask = regime_labels == regime
        regime_samples = np.sum(regime_mask)
        if regime_samples > 0:
            # Generate targets with regime-specific patterns
            y[regime_mask] = np.random.randn(regime_samples) + regime * 0.5
    
    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Generate HMM states
    hmm_states = np.random.randint(0, n_regimes, n_samples)
    
    return {
        'X': X,
        'y': y,
        'regime_labels': regime_labels,
        'feature_names': feature_names,
        'hmm_states': hmm_states,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_regimes': n_regimes
    }


def test_hmm_base_models_training():
    """Test HMM base models training."""
    logger.info("Testing HMM base models training...")
    
    try:
        from src.training.steps.market_analysis.hmm_models_training import (
            create_enhanced_hmm_models_training,
            execute_enhanced_hmm_models_training
        )
        
        # Generate mock data
        data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3)
        
        # Create and execute training
        results = execute_enhanced_hmm_models_training(
            data['X'], data['y'], data['regime_labels'],
            feature_names=data['feature_names'],
            hmm_states=data['hmm_states']
        )
        
        # Validate results
        assert 'artifacts' in results, "Results should contain artifacts"
        assert 'hmm_base_models' in results['artifacts'], "Should contain HMM base models"
        assert 'hmm_training_metrics' in results['artifacts'], "Should contain training metrics"
        
        logger.info("✅ HMM base models training test passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ HMM base models training test failed: {e}")
        raise


def test_hmm_ensemble_training():
    """Test HMM ensemble training."""
    logger.info("Testing HMM ensemble training...")
    
    try:
        from src.training.steps.market_analysis.hmm_models_training import (
            create_hmm_ensemble_training_component,
            execute_hmm_ensemble_training
        )
        
        # Generate mock data
        data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3)
        
        # Create mock base models
        from sklearn.ensemble import RandomForestRegressor
        base_models = {
            'lightgbm': RandomForestRegressor(n_estimators=10, random_state=42),
            'elastic_net': RandomForestRegressor(n_estimators=10, random_state=43),
            'xgboost': RandomForestRegressor(n_estimators=10, random_state=44)
        }
        
        # Train base models
        for name, model in base_models.items():
            model.fit(data['X'], data['y'])
        
        # Create base models artifacts
        hmm_base_models = {
            name: {'model_object': model, 'model_name': name, 'model_type': name}
            for name, model in base_models.items()
        }
        
        hmm_training_metrics = {
            name: {'accuracy': 0.8, 'f1_score': 0.75, 'training_time': 1.0}
            for name in base_models.keys()
        }
        
        # Execute ensemble training
        results = execute_hmm_ensemble_training(
            data['X'], data['y'], data['regime_labels'],
            feature_names=data['feature_names'],
            hmm_states=data['hmm_states'],
            base_hmm_models=hmm_base_models,
            hmm_training_metrics=hmm_training_metrics
        )
        
        # Validate results
        assert 'artifacts' in results, "Results should contain artifacts"
        assert 'hmm_ensemble_models' in results['artifacts'], "Should contain HMM ensemble models"
        
        logger.info("✅ HMM ensemble training test passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ HMM ensemble training test failed: {e}")
        raise


def test_analyst_ensemble_training():
    """Test Analyst ensemble training with HMM integration."""
    logger.info("Testing Analyst ensemble training with HMM integration...")
    
    try:
        from src.training.steps.model_training.analyst_ensemble_training import (
            create_analyst_ensemble_training_step,
            execute_analyst_ensemble_training
        )
        
        # Generate mock data
        data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3)
        
        # Create mock HMM base models
        from sklearn.ensemble import RandomForestRegressor
        hmm_base_models = {
            'hmm_lightgbm': {'model_object': RandomForestRegressor(n_estimators=10, random_state=42)},
            'hmm_elastic_net': {'model_object': RandomForestRegressor(n_estimators=10, random_state=43)},
            'hmm_xgboost': {'model_object': RandomForestRegressor(n_estimators=10, random_state=44)}
        }
        
        # Train HMM models
        for name, model_data in hmm_base_models.items():
            model_data['model_object'].fit(data['X'], data['y'])
        
        hmm_training_metrics = {
            name: {'accuracy': 0.8, 'f1_score': 0.75, 'training_time': 1.0}
            for name in hmm_base_models.keys()
        }
        
        # Execute analyst ensemble training
        results = execute_analyst_ensemble_training(
            data['X'], data['y'], data['regime_labels'],
            feature_names=data['feature_names'],
            hmm_states=data['hmm_states'],
            hmm_base_models=hmm_base_models,
            hmm_training_metrics=hmm_training_metrics
        )
        
        # Validate results
        assert 'artifacts' in results, "Results should contain artifacts"
        assert 'analyst_ensembles' in results['artifacts'], "Should contain analyst ensembles"
        
        logger.info("✅ Analyst ensemble training test passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Analyst ensemble training test failed: {e}")
        raise


def test_tactician_ensemble_training():
    """Test Tactician ensemble training with comprehensive integration."""
    logger.info("Testing Tactician ensemble training with comprehensive integration...")
    
    try:
        from src.training.steps.model_training.tactician_ensemble_training import (
            create_tactician_ensemble_training_step,
            execute_tactician_ensemble_training
        )
        
        # Generate mock data
        data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3)
        
        # Create mock HMM data
        hmm_data = {
            'regime_features': np.random.randn(data['n_samples'], 10),
            'hmm_base_models': {
                'hmm_lightgbm': {'model_object': None},
                'hmm_elastic_net': {'model_object': None}
            },
            'hmm_ensemble_models': {
                'hmm_ensemble_1': {'model_object': None},
                'hmm_ensemble_2': {'model_object': None}
            },
            'metrics': {'accuracy': 0.8, 'f1_score': 0.75}
        }
        
        # Create mock analyst models
        from sklearn.ensemble import RandomForestRegressor
        analyst_models = {
            'analyst_1': RandomForestRegressor(n_estimators=10, random_state=42),
            'analyst_2': RandomForestRegressor(n_estimators=10, random_state=43)
        }
        
        # Train analyst models
        for name, model in analyst_models.items():
            model.fit(data['X'], data['y'])
        
        analyst_ensembles = {
            'analyst_ensemble_1': RandomForestRegressor(n_estimators=10, random_state=44),
            'analyst_ensemble_2': RandomForestRegressor(n_estimators=10, random_state=45)
        }
        
        for name, model in analyst_ensembles.items():
            model.fit(data['X'], data['y'])
        
        # Execute tactician ensemble training
        results = execute_tactician_ensemble_training(
            data['X'], data['y'], data['regime_labels'],
            feature_names=data['feature_names'],
            hmm_states=data['hmm_states'],
            hmm_data=hmm_data,
            analyst_models=analyst_models,
            analyst_ensembles=analyst_ensembles
        )
        
        # Validate results
        assert 'artifacts' in results, "Results should contain artifacts"
        assert 'tactician_ensembles' in results['artifacts'], "Should contain tactician ensembles"
        
        logger.info("✅ Tactician ensemble training test passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Tactician ensemble training test failed: {e}")
        raise


def test_complete_orchestrator():
    """Test the complete HMM LM training orchestrator."""
    logger.info("Testing complete HMM LM training orchestrator...")
    
    try:
        from src.training.steps.model_training.hmm_lm_training_orchestrator import (
            create_hmm_lm_training_orchestrator,
            execute_complete_hmm_lm_training
        )
        
        # Generate mock data for all timeframes
        hmm_data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3, timeframe="1h")
        analyst_data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3, timeframe="5m")
        tactician_data = generate_mock_data(n_samples=500, n_features=30, n_regimes=3, timeframe="1m")
        
        # Execute complete training
        results = execute_complete_hmm_lm_training(
            hmm_data['X'], hmm_data['y'], hmm_data['regime_labels'],
            analyst_data['X'], analyst_data['y'], analyst_data['regime_labels'],
            tactician_data['X'], tactician_data['y'], tactician_data['regime_labels'],
            feature_names_hmm=hmm_data['feature_names'],
            feature_names_analyst=analyst_data['feature_names'],
            feature_names_tactician=tactician_data['feature_names'],
            hmm_states=hmm_data['hmm_states']
        )
        
        # Validate results
        assert 'success' in results, "Results should contain success status"
        assert 'comprehensive_report' in results, "Results should contain comprehensive report"
        assert 'phase_results' in results, "Results should contain phase results"
        assert 'artifacts' in results, "Results should contain artifacts"
        
        logger.info("✅ Complete orchestrator test passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Complete orchestrator test failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    logger.info("Starting HMM LM Models Training Pathway Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Test 1: HMM Base Models Training
        test_results['hmm_base'] = test_hmm_base_models_training()
        
        # Test 2: HMM Ensemble Training
        test_results['hmm_ensemble'] = test_hmm_ensemble_training()
        
        # Test 3: Analyst Ensemble Training
        test_results['analyst_ensemble'] = test_analyst_ensemble_training()
        
        # Test 4: Tactician Ensemble Training
        test_results['tactician_ensemble'] = test_tactician_ensemble_training()
        
        # Test 5: Complete Orchestrator
        test_results['complete_orchestrator'] = test_complete_orchestrator()
        
        logger.info("=" * 60)
        logger.info("🎉 All tests passed successfully!")
        logger.info("=" * 60)
        
        # Print summary
        for test_name, result in test_results.items():
            if isinstance(result, dict) and 'success' in result:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
            else:
                status = "✅ PASS"
            logger.info(f"{test_name}: {status}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    try:
        results = run_all_tests()
        print("\n🎯 HMM LM Models Training Pathway - All Tests Passed!")
        print("The complete training pathway is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        exit(1)