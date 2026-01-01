#!/usr/bin/env python3
"""Test specialist independence pattern implementation."""

import sys
sys.path.insert(0, '.')

def test_specialist():
    """Test that specialist has independence pattern."""
    try:
        from src.training.steps.market_analysis.ml_momentum_persistence_step import MLMomentumPersistenceStep
        
        # Test instantiation
        specialist = MLMomentumPersistenceStep()
        print("✅ Specialist instantiated successfully")
        
        # Test diagnostics method
        if hasattr(specialist, 'run_diagnostics'):
            print("✅ run_diagnostics method available")
        else:
            print("❌ run_diagnostics method missing")
            return False
            
        # Test mixin methods
        if hasattr(specialist, 'run_self_diagnostics'):
            print("✅ run_self_diagnostics method available")
        else:
            print("❌ run_self_diagnostics method missing")
            return False
            
        if hasattr(specialist, '_load_self_artifacts'):
            print("✅ _load_self_artifacts method available")
        else:
            print("❌ _load_self_artifacts method missing")
            return False
            
        print("✅ Independence pattern successfully implemented!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_specialist()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
