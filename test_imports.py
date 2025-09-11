"""
Simple import test for the refactored feature selection framework.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test importing the main framework
        from src.training.utils.feature_selection import FeatureSelectionFramework
        print("✅ Successfully imported FeatureSelectionFramework")
        
        # Test importing individual components
        from src.training.utils.feature_selection import (
            BaseFeatureSelectionFramework,
            DataValidator,
            MRMRSelector,
            LassoStabilitySelector,
            CorrelationBasedFilter,
            RecursiveFeatureEliminator,
            FeatureImportanceRanker,
            StabilityAnalyzer,
            PerformanceMonitor,
            QualityMetricsCalculator,
            TemporalAnalyzer,
            CausalAnalyzer
        )
        print("✅ Successfully imported all individual components")
        
        # Test that the main class can be instantiated
        config = {'random_state': 42}
        framework = FeatureSelectionFramework(config)
        print("✅ Successfully instantiated FeatureSelectionFramework")
        
        # Test that expected methods exist
        expected_methods = [
            'run_comprehensive_feature_selection',
            'get_model_target_features',
            'get_optimization_stats',
            'check_system_requirements'
        ]
        
        for method in expected_methods:
            if hasattr(framework, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        print("✅ All expected methods exist")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! The refactoring is working correctly.")
    else:
        print("\n⚠️ Some imports failed. Please check the module structure.")
    sys.exit(0 if success else 1)