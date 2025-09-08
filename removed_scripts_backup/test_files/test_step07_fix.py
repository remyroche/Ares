#!/usr/bin/env python3
"""
Test script to verify that Step7EnhancedMatrixOperations can be initialized properly
after fixing the logging decorator bug.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_step07_initialization():
    """Test that Step7EnhancedMatrixOperations can be initialized without coroutine error."""
    try:
        # First initialize the logging system
        from src.utils.logger import setup_logging, system_logger
        print("🔧 Initializing logging system...")
        setup_logging()

        from src.training.steps.market_analysis.step07_enhanced_matrix_operations import Step7EnhancedMatrixOperations

        # Test configuration
        config = {
            'step07_enhanced_matrix_operations': {
                'output_dir': 'test_output',
                'target_features': 200,
                'removal_fraction': 0.33,
                'enable_regime_selection': True,
                'enable_shap_filtering': True
            }
        }

        print("🔍 Testing Step7EnhancedMatrixOperations initialization...")

        # This should work now without the coroutine error
        step07_instance = Step7EnhancedMatrixOperations(config=config)

        print("✅ Step7EnhancedMatrixOperations initialized successfully!")
        print(f"   Target features: {step07_instance.target_features}")
        print(f"   Output dir: {step07_instance.output_dir}")
        print(f"   Enable regime selection: {step07_instance.enable_regime_selection}")

        return True

    except Exception as e:
        print(f"❌ Step7EnhancedMatrixOperations initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Step07 fix...")
    success = test_step07_initialization()

    if success:
        print("\n🎉 Step07 fix verified successfully!")
        sys.exit(0)
    else:
        print("\n💥 Step07 fix verification failed!")
        sys.exit(1)
