#!/usr/bin/env python3
"""
Simple test for enhanced directional signal structure.

This script tests the new directional signal functionality without requiring
full dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing Imports")
    print("=" * 30)
    
    try:
        # Test directional signal structure import
        from src.training.steps.model_training.directional_signal_structure import (
            DirectionalSignalArray, DirectionalSignal, SignalDirection
        )
        print("✅ Directional signal structure imports successful")
        
        # Test analyst training import
        from src.training.steps.model_training.analyst_models_training_refactored import (
            AnalystModelsTrainingStepRefactored
        )
        print("✅ Analyst training imports successful")
        
        # Test tactician training import
        from src.training.steps.model_training.tactician_models_training_refactored import (
            TacticianModelsTrainingStepRefactored
        )
        print("✅ Tactician training imports successful")
        
        # Test ensemble training import
        from src.training.steps.model_training.tactician_ensemble_training import (
            TacticianEnsembleTrainingStep
        )
        print("✅ Ensemble training imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_signal_direction_enum():
    """Test the SignalDirection enum."""
    print("\n🧪 Testing SignalDirection Enum")
    print("=" * 30)
    
    try:
        from src.training.steps.model_training.directional_signal_structure import SignalDirection
        
        # Test enum values
        assert SignalDirection.LONG.value == "long"
        assert SignalDirection.SHORT.value == "short"
        assert SignalDirection.NEUTRAL.value == "neutral"
        
        print("✅ SignalDirection enum values correct")
        
        # Test enum comparison
        assert SignalDirection.LONG != SignalDirection.SHORT
        assert SignalDirection.LONG != SignalDirection.NEUTRAL
        
        print("✅ SignalDirection enum comparison works")
        
        return True
        
    except Exception as e:
        print(f"❌ SignalDirection enum test failed: {e}")
        return False


def test_directional_signal_creation():
    """Test creating DirectionalSignal objects."""
    print("\n🧪 Testing DirectionalSignal Creation")
    print("=" * 30)
    
    try:
        from src.training.steps.model_training.directional_signal_structure import (
            DirectionalSignal, SignalDirection
        )
        
        # Test creating a valid signal
        signal = DirectionalSignal(
            is_active=True,
            direction=SignalDirection.LONG,
            confidence=0.8,
            strength=0.7,
            expected_return=0.02,
            risk_score=0.3,
            duration_minutes=30,
            urgency=0.6
        )
        
        assert signal.is_active == True
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.8
        assert signal.strength == 0.7
        assert signal.expected_return == 0.02
        assert signal.risk_score == 0.3
        assert signal.duration_minutes == 30
        assert signal.urgency == 0.6
        
        print("✅ DirectionalSignal creation successful")
        
        # Test validation
        try:
            invalid_signal = DirectionalSignal(
                is_active=True,
                direction=SignalDirection.LONG,
                confidence=1.5,  # Invalid: > 1.0
                strength=0.7,
                expected_return=0.02,
                risk_score=0.3,
                duration_minutes=30,
                urgency=0.6
            )
            print("❌ Should have failed with invalid confidence")
            return False
        except ValueError:
            print("✅ Validation correctly caught invalid confidence")
        
        return True
        
    except Exception as e:
        print(f"❌ DirectionalSignal creation test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure")
    print("=" * 30)
    
    required_files = [
        "src/training/steps/model_training/directional_signal_structure.py",
        "src/training/steps/model_training/analyst_models_training_refactored.py",
        "src/training/steps/model_training/tactician_models_training_refactored.py",
        "src/training/steps/model_training/tactician_ensemble_training.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Simple Directional Signal Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_signal_direction_enum,
        test_directional_signal_creation
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
    
    print("\n" + "=" * 50)
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