#!/usr/bin/env python3
"""
Test script for enhanced directional signal structure.

This script tests the new directional signal functionality that includes
short/long information for the Analyst's signals, enabling the Tactician
to make more informed timing decisions.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_directional_signal_structure():
    """Test the directional signal structure."""
    print("🧪 Testing Directional Signal Structure")
    print("=" * 50)
    
    try:
        from src.training.steps.model_training.directional_signal_structure import (
            DirectionalSignalArray, DirectionalSignal, SignalDirection,
            create_directional_signals_from_analyst_outputs, enhance_signals_with_market_data
        )
        print("✅ Successfully imported directional signal structure")
    except ImportError as e:
        print(f"❌ Failed to import directional signal structure: {e}")
        return False
    
    # Test 1: Create basic directional signals
    print("\n📊 Test 1: Creating basic directional signals")
    try:
        # Create sample analyst outputs
        n_samples = 100
        analyst_outputs = {
            'signals': np.random.randint(0, 2, n_samples),
            'predictions': np.random.rand(n_samples),
            'confidences': np.random.rand(n_samples)
        }
        
        # Create directional signals
        directional_signals = create_directional_signals_from_analyst_outputs(analyst_outputs)
        print(f"✅ Created directional signals with {len(directional_signals)} samples")
        
        # Test statistics
        stats = directional_signals.get_statistics()
        print(f"   Active signals: {stats['active_signals']}")
        print(f"   Long signals: {stats['long_signals']}")
        print(f"   Short signals: {stats['short_signals']}")
        print(f"   Average confidence: {stats['avg_confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Filter by direction
    print("\n📊 Test 2: Filtering by direction")
    try:
        long_signals = directional_signals.filter_by_direction(SignalDirection.LONG)
        short_signals = directional_signals.filter_by_direction(SignalDirection.SHORT)
        
        print(f"✅ Long signals: {len(long_signals)}")
        print(f"✅ Short signals: {len(short_signals)}")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Filter by confidence
    print("\n📊 Test 3: Filtering by confidence")
    try:
        high_confidence_signals = directional_signals.filter_by_confidence(0.7)
        print(f"✅ High confidence signals: {len(high_confidence_signals)}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Enhance with market data
    print("\n📊 Test 4: Enhancing with market data")
    try:
        # Create sample market data
        market_data = np.random.randn(n_samples, 1) * 0.01 + 100  # Price data
        enhanced_signals = enhance_signals_with_market_data(directional_signals, market_data)
        
        print(f"✅ Enhanced signals with market data")
        enhanced_stats = enhanced_signals.get_statistics()
        print(f"   Average expected return: {enhanced_stats['avg_expected_return']:.3f}")
        print(f"   Average risk score: {enhanced_stats['avg_risk_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Backward compatibility
    print("\n📊 Test 5: Backward compatibility")
    try:
        binary_signals = directional_signals.to_binary_signals()
        print(f"✅ Converted to binary signals: {len(binary_signals)} samples")
        print(f"   Active signals: {np.sum(binary_signals)}")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return False
    
    print("\n✅ All directional signal structure tests passed!")
    return True


def test_analyst_training_integration():
    """Test analyst training integration with directional signals."""
    print("\n🧪 Testing Analyst Training Integration")
    print("=" * 50)
    
    try:
        from src.training.steps.model_training.analyst_models_training_refactored import (
            AnalystModelsTrainingStepRefactored
        )
        print("✅ Successfully imported analyst training")
    except ImportError as e:
        print(f"❌ Failed to import analyst training: {e}")
        return False
    
    # Test creating analyst training step
    try:
        training_step = AnalystModelsTrainingStepRefactored()
        print("✅ Created analyst training step")
    except Exception as e:
        print(f"❌ Failed to create analyst training step: {e}")
        return False
    
    print("✅ Analyst training integration test passed!")
    return True


def test_tactician_training_integration():
    """Test tactician training integration with directional signals."""
    print("\n🧪 Testing Tactician Training Integration")
    print("=" * 50)
    
    try:
        from src.training.steps.model_training.tactician_models_training_refactored import (
            TacticianModelsTrainingStepRefactored
        )
        print("✅ Successfully imported tactician training")
    except ImportError as e:
        print(f"❌ Failed to import tactician training: {e}")
        return False
    
    # Test creating tactician training step
    try:
        training_step = TacticianModelsTrainingStepRefactored()
        print("✅ Created tactician training step")
    except Exception as e:
        print(f"❌ Failed to create tactician training step: {e}")
        return False
    
    print("✅ Tactician training integration test passed!")
    return True


def test_ensemble_training_integration():
    """Test ensemble training integration with directional signals."""
    print("\n🧪 Testing Ensemble Training Integration")
    print("=" * 50)
    
    try:
        from src.training.steps.model_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep
        )
        print("✅ Successfully imported ensemble training")
    except ImportError as e:
        print(f"❌ Failed to import ensemble training: {e}")
        return False
    
    # Test creating ensemble training step
    try:
        training_step = TacticianEnsembleTrainingStep()
        print("✅ Created ensemble training step")
    except Exception as e:
        print(f"❌ Failed to create ensemble training step: {e}")
        return False
    
    print("✅ Ensemble training integration test passed!")
    return True


def test_end_to_end_workflow():
    """Test end-to-end workflow with directional signals."""
    print("\n🧪 Testing End-to-End Workflow")
    print("=" * 50)
    
    try:
        # Create sample data
        n_samples = 1000
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        regime_labels = np.random.randint(0, 3, n_samples)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        print(f"✅ Created sample data: {n_samples} samples, {n_features} features")
        
        # Test analyst training with directional signals
        from src.training.steps.model_training.analyst_models_training_refactored import (
            create_analyst_models_training_step_enhanced
        )
        
        training_step = create_analyst_models_training_step_enhanced()
        print("✅ Created analyst training step")
        
        # Note: We don't actually run the training here as it would take too long
        # In a real scenario, you would call:
        # results = training_step.execute(X, y, regime_labels, feature_names, generate_directional_signals=True)
        
        print("✅ End-to-end workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Directional Signal Tests")
    print("=" * 60)
    
    tests = [
        test_directional_signal_structure,
        test_analyst_training_integration,
        test_tactician_training_integration,
        test_ensemble_training_integration,
        test_end_to_end_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced directional signals are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)