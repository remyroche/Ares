#!/usr/bin/env python3
"""
Test script to verify that S/R optimization pipeline can use step07 feature selection
after fixing the logging decorator bug.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sr_step07_integration():
    """Test that S/R optimization can use step07 feature selection without coroutine error."""
    try:
        # Initialize logging first
        from src.utils.logger import setup_logging
        print("🔧 Initializing logging system...")
        setup_logging()

        from src.training.steps.data_collection.data_preparation.step02_5_sr_optimization import SROptimizationStep

        # Create minimal test data
        print("📊 Creating test data...")
        np.random.seed(42)

        # Create sample features (smaller for testing)
        n_samples = 1000
        n_features = 50
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Create sample targets
        y_sr_detection = pd.Series(np.random.randint(0, 3, n_samples))
        y_strength = pd.Series(np.random.rand(n_samples))

        # Test configuration
        config = {
            'feature_selection': {
                'enable_mi_shap_preselection': True
            },
            'step07_enhanced_matrix_operations': {
                'output_dir': 'test_output',
                'target_features': 30,
                'removal_fraction': 0.33,
                'enable_regime_selection': True,
                'enable_shap_filtering': True
            }
        }

        print("🔍 Testing S/R optimization with step07 feature selection...")

        # Initialize S/R optimization step
        sr_step = SROptimizationStep(config=config)

        # Test the step07 feature selection method directly
        print("🧪 Testing _apply_step07_feature_selection method...")
        X_selected, selection_info = sr_step._apply_step07_feature_selection(
            X, y_sr_detection
        )

        print("✅ Step07 feature selection completed successfully!")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Selection method: {selection_info.get('method', 'unknown')}")
        print(f"   Computation time: {selection_info.get('computation_time', 0):.3f}s")

        return True

    except Exception as e:
        print(f"❌ S/R optimization with step07 integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing S/R optimization pipeline with step07 integration...")
    success = test_sr_step07_integration()

    if success:
        print("\n🎉 S/R optimization with step07 integration verified successfully!")
        sys.exit(0)
    else:
        print("\n💥 S/R optimization with step07 integration verification failed!")
        sys.exit(1)
