#!/usr/bin/env python3
"""
Test script for Tactician & Analyst integration with whole dataset training.

This script tests the updated implementation where:
1. Tactician is trained on the whole dataset
2. Analyst OOF outputs (p_trade, u_trade, q_trade) are used as features
3. Sample weights are calculated based on Analyst confidence
4. Final parameters optimization handles merged inputs
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

async def test_analyst_oof_generation():
    """Test Analyst OOF prediction generation."""
    print("🧪 Testing Analyst OOF prediction generation...")
    
    try:
        from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        training_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })
        training_data['target'] = np.random.randn(n_samples)
        
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        target_columns = ['target']
        sample_weight = np.ones(n_samples)
        
        # Initialize trainer
        trainer = AnalystModelsTrainingStep()
        
        # Test OOF generation
        oof_predictions = await trainer._generate_oof_predictions(
            training_data, feature_columns, target_columns, sample_weight
        )
        
        # Verify OOF predictions structure
        assert 'p_trade' in oof_predictions
        assert 'u_trade' in oof_predictions
        assert 'q_trade' in oof_predictions
        assert len(oof_predictions['p_trade']) == n_samples
        
        print(f"✅ Analyst OOF generation test passed")
        print(f"   • p_trade range: {min(oof_predictions['p_trade']):.3f} - {max(oof_predictions['p_trade']):.3f}")
        print(f"   • u_trade range: {min(oof_predictions['u_trade']):.3f} - {max(oof_predictions['u_trade']):.3f}")
        print(f"   • q_trade range: {min(oof_predictions['q_trade']):.3f} - {max(oof_predictions['q_trade']):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyst OOF generation test failed: {e}")
        return False

async def test_tactician_analyst_integration():
    """Test Tactician training with Analyst OOF features."""
    print("🧪 Testing Tactician & Analyst integration...")
    
    try:
        from src.training.steps.models_training.tactician_models_training import TacticianModelsTrainingStep
        
        # Create sample data with Analyst OOF outputs
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        training_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })
        training_data['target'] = np.random.randn(n_samples)
        
        # Add mock Analyst OOF outputs
        training_data['analyst_p_trade'] = np.random.uniform(0, 1, n_samples)
        training_data['analyst_u_trade'] = np.random.uniform(-1, 1, n_samples)
        training_data['analyst_q_trade'] = np.random.uniform(0, 1, n_samples)
        
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        target_columns = ['target']
        
        # Initialize trainer
        trainer = TacticianModelsTrainingStep()
        
        # Test Analyst OOF feature addition
        analyst_features = await trainer._add_analyst_oof_features(training_data)
        
        # Verify features were added
        assert len(analyst_features) > 0
        assert any('analyst' in feat for feat in analyst_features)
        
        # Test weight calculation
        sample_weights = await trainer._calculate_analyst_weights(training_data, w_min=0.2)
        
        # Verify weights were calculated
        assert sample_weights is not None
        assert len(sample_weights) == n_samples
        assert all(w > 0 for w in sample_weights)
        
        print(f"✅ Tactician & Analyst integration test passed")
        print(f"   • Analyst features added: {len(analyst_features)}")
        print(f"   • Sample weights range: {min(sample_weights):.3f} - {max(sample_weights):.3f}")
        print(f"   • Sample weights mean: {np.mean(sample_weights):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tactician & Analyst integration test failed: {e}")
        return False

async def test_final_parameters_optimization():
    """Test final parameters optimization with merged inputs."""
    print("🧪 Testing final parameters optimization...")
    
    try:
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        
        # Initialize optimizer
        config = {
            'n_trials': 5,  # Small number for testing
            'timeout': 30,
            'study_name': 'test_optimization'
        }
        
        optimizer = FinalParametersOptimizer(config)
        
        # Check if new categories are included
        assert 'tactician_analyst_integration' in optimizer.categories
        assert 'analyst_oof_weights' in optimizer.categories
        assert 'merged_feature_importance' in optimizer.categories
        
        # Check if search spaces are defined
        assert 'tactician_analyst_integration' in optimizer.default_search_spaces
        assert 'analyst_oof_weights' in optimizer.default_search_spaces
        assert 'merged_feature_importance' in optimizer.default_search_spaces
        
        print(f"✅ Final parameters optimization test passed")
        print(f"   • Categories: {len(optimizer.categories)}")
        print(f"   • New categories added: tactician_analyst_integration, analyst_oof_weights, merged_feature_importance")
        
        return True
        
    except Exception as e:
        print(f"❌ Final parameters optimization test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Tactician & Analyst integration tests...")
    print("=" * 60)
    
    tests = [
        ("Analyst OOF Generation", test_analyst_oof_generation),
        ("Tactician & Analyst Integration", test_tactician_analyst_integration),
        ("Final Parameters Optimization", test_final_parameters_optimization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Tactician & Analyst integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)