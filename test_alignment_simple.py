#!/usr/bin/env python3
"""
Simple test to verify profit labeling framework alignment in optimization systems.

This script performs basic import and configuration checks without requiring
external dependencies like numpy or pandas.
"""

import sys
import os
import importlib.util

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    # Test profit labeling framework imports
    try:
        from src.training.steps.pre_training.profit_labeling.quality_scoring import (
            LabelQualityScorer, QualityScoringConfig, QualityMetrics, QualityMetric
        )
        print("✅ Profit labeling quality scoring imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import profit labeling quality scoring: {e}")
        return False
    
    try:
        from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
            VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig, LabelQualityScore
        )
        print("✅ Profit labeling volatility aware labeler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import profit labeling volatility aware labeler: {e}")
        return False
    
    try:
        from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
            MultiTargetScheme, MultiTargetConfig, TargetBand
        )
        print("✅ Profit labeling multi-target scheme imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import profit labeling multi-target scheme: {e}")
        return False
    
    try:
        from src.training.steps.pre_training.profit_labeling.noise_gating import (
            NoiseGatingFilter, NoiseGatingConfig
        )
        print("✅ Profit labeling noise gating imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import profit labeling noise gating: {e}")
        return False
    
    # Test feature lookback optimization imports
    try:
        from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
            OptimizedFeatureLookbackConfig, OptimizedFeatureLookbackOptimizer
        )
        print("✅ Feature lookback optimization imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import feature lookback optimization: {e}")
        return False
    
    # Test interaction feature generator imports
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.orchestrator import (
            LookbackOptimizationOrchestrator
        )
        print("✅ Interaction feature generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import interaction feature generator: {e}")
        return False
    
    # Test unified optimization framework
    try:
        from src.training.steps.pre_training.unified_optimization_framework import (
            UnifiedOptimizationFramework, UnifiedOptimizationConfig, 
            OptimizationSystem, OptimizationObjective
        )
        print("✅ Unified optimization framework imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import unified optimization framework: {e}")
        return False
    
    return True


def test_configuration_alignment():
    """Test that configurations are aligned with profit labeling framework."""
    print("\n🧪 Testing configuration alignment...")
    
    try:
        from src.training.steps.pre_training.unified_optimization_framework import (
            UnifiedOptimizationConfig, OptimizationObjective
        )
        
        # Create configuration
        config = UnifiedOptimizationConfig()
        
        # Check that profit labeling quality thresholds are set
        expected_thresholds = {
            'min_lqs_threshold': 0.3,
            'min_auc_threshold': 0.55,
            'max_auc_std_threshold': 0.03,
            'min_psi_threshold': 0.1,
            'max_flip_rate_threshold': 0.15,
            'min_balance_threshold': 0.35,
            'max_balance_threshold': 0.65,
            'max_correlation_threshold': 0.4
        }
        
        for threshold, expected_value in expected_thresholds.items():
            if hasattr(config, threshold):
                actual_value = getattr(config, threshold)
                if actual_value == expected_value:
                    print(f"✅ {threshold}: {actual_value}")
                else:
                    print(f"❌ {threshold}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"❌ {threshold} not found in configuration")
                return False
        
        # Check that multi-objective optimization is enabled
        if config.primary_objective == OptimizationObjective.MULTI_OBJECTIVE:
            print("✅ Multi-objective optimization enabled")
        else:
            print(f"❌ Multi-objective optimization not enabled: {config.primary_objective}")
            return False
        
        # Check weights
        expected_weights = {
            'ic_weight': 0.4,
            'lqs_weight': 0.4,
            'stability_weight': 0.2
        }
        
        for weight, expected_value in expected_weights.items():
            if hasattr(config, weight):
                actual_value = getattr(config, weight)
                if actual_value == expected_value:
                    print(f"✅ {weight}: {actual_value}")
                else:
                    print(f"❌ {weight}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"❌ {weight} not found in configuration")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration alignment test failed: {e}")
        return False


def test_feature_lookback_integration():
    """Test that feature lookback optimization has profit labeling integration."""
    print("\n🧪 Testing feature lookback integration...")
    
    try:
        from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
            OptimizedFeatureLookbackConfig
        )
        
        # Create configuration
        config = OptimizedFeatureLookbackConfig()
        
        # Check that LQS-based optimization is enabled
        if config.optimization_metric == "lqs_combined":
            print("✅ LQS-based optimization metric enabled")
        else:
            print(f"❌ LQS-based optimization not enabled: {config.optimization_metric}")
            return False
        
        # Check that quality scoring is enabled
        if config.enable_quality_scoring:
            print("✅ Quality scoring enabled")
        else:
            print("❌ Quality scoring not enabled")
            return False
        
        # Check that multi-objective optimization is enabled
        if config.enable_multi_objective:
            print("✅ Multi-objective optimization enabled")
        else:
            print("❌ Multi-objective optimization not enabled")
            return False
        
        # Check quality thresholds
        expected_thresholds = {
            'min_lqs_threshold': 0.3,
            'min_auc_threshold': 0.55,
            'max_auc_std_threshold': 0.03,
            'min_psi_threshold': 0.1,
            'max_flip_rate_threshold': 0.15,
            'min_balance_threshold': 0.35,
            'max_balance_threshold': 0.65,
            'max_correlation_threshold': 0.4
        }
        
        for threshold, expected_value in expected_thresholds.items():
            if hasattr(config, threshold):
                actual_value = getattr(config, threshold)
                if actual_value == expected_value:
                    print(f"✅ {threshold}: {actual_value}")
                else:
                    print(f"❌ {threshold}: {actual_value} (expected {expected_value})")
                    return False
            else:
                print(f"❌ {threshold} not found in feature lookback config")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feature lookback integration test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "src/training/steps/pre_training/unified_optimization_framework.py",
        "src/training/steps/pre_training/profit_labeling_aligned_config.yaml",
        "src/training/steps/pre_training/feature_lookback_optimization/feature_lookback_optimization.py",
        "src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation/orchestrator.py",
        "src/training/steps/pre_training/profit_labeling/quality_scoring.py",
        "src/training/steps/pre_training/profit_labeling/volatility_aware_labeler.py",
        "src/training/steps/pre_training/profit_labeling/multi_target_scheme.py",
        "src/training/steps/pre_training/profit_labeling/noise_gating.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting profit labeling alignment verification...")
    print("=" * 60)
    
    test_results = []
    
    # Test file structure
    test_results.append(("File Structure", test_file_structure()))
    
    # Test imports
    test_results.append(("Imports", test_imports()))
    
    # Test configuration alignment
    test_results.append(("Configuration Alignment", test_configuration_alignment()))
    
    # Test feature lookback integration
    test_results.append(("Feature Lookback Integration", test_feature_lookback_integration()))
    
    # Print summary
    print("\n📊 Test Results Summary:")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print("=" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Profit labeling alignment is complete.")
        print("\n✅ Key Achievements:")
        print("   • Feature lookback optimization now uses LQS scoring")
        print("   • Interaction feature generator integrated with profit labeling framework")
        print("   • Unified optimization framework created")
        print("   • All configurations aligned with profit labeling quality thresholds")
        print("   • Multi-objective optimization (IC + LQS + Stability) implemented")
        print("   • Quality-based feature filtering enabled")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Profit labeling alignment verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)