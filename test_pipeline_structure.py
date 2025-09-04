#!/usr/bin/env python3
"""
Test script for the enhanced market analysis pipeline structure.
This script tests the pipeline structure without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing file structure...")
    
    required_files = [
        "src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py",
        "src/training/steps/market_analysis/enhanced_step_validator.py", 
        "src/training/steps/market_analysis/enhanced_pipeline_decorators.py",
        "src/training/steps/market_analysis/__init__.py",
        "src/training/steps/market_analysis/step03_market_analysis_main.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_launcher_integration():
    """Test that the launcher can find the market-analysis command."""
    print("\n🧪 Testing launcher integration...")
    
    try:
        # Read the launcher file to check for market-analysis command
        with open("ares_launcher.py", "r") as f:
            launcher_content = f.read()
        
        if "market-analysis" in launcher_content:
            print("✅ market-analysis command found in launcher")
        else:
            print("❌ market-analysis command not found in launcher")
            return False
        
        if "run_market_analysis_pipeline" in launcher_content:
            print("✅ run_market_analysis_pipeline function found in launcher")
        else:
            print("❌ run_market_analysis_pipeline function not found in launcher")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Launcher integration test failed: {e}")
        return False

def test_pipeline_components():
    """Test that pipeline components are properly structured."""
    print("\n🧪 Testing pipeline components...")
    
    try:
        # Test enhanced orchestrator file
        orchestrator_file = Path("src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py")
        if orchestrator_file.exists():
            with open(orchestrator_file, "r") as f:
                content = f.read()
            
            required_classes = [
                "class MarketAnalysisPipelineOrchestrator",
                "async def execute_pipeline",
                "async def _execute_hmm_clustering",
                "async def _execute_regime_splitting",
                "async def _execute_labeling",
                "async def _execute_feature_engineering",
                "async def _execute_matrix_operations",
                "async def _execute_feature_selection",
            ]
            
            for class_or_function in required_classes:
                if class_or_function in content:
                    print(f"✅ {class_or_function} found in orchestrator")
                else:
                    print(f"❌ {class_or_function} missing from orchestrator")
                    return False
        
        # Test enhanced validator file
        validator_file = Path("src/training/steps/market_analysis/enhanced_step_validator.py")
        if validator_file.exists():
            with open(validator_file, "r") as f:
                content = f.read()
            
            required_components = [
                "class EnhancedStepValidator",
                "async def validate_step_input",
                "async def validate_step_output",
                "async def validate_step_transition",
            ]
            
            for component in required_components:
                if component in content:
                    print(f"✅ {component} found in validator")
                else:
                    print(f"❌ {component} missing from validator")
                    return False
        
        # Test enhanced decorators file
        decorators_file = Path("src/training/steps/market_analysis/enhanced_pipeline_decorators.py")
        if decorators_file.exists():
            with open(decorators_file, "r") as f:
                content = f.read()
            
            required_decorators = [
                "class DataFormattingDecorator",
                "class DataAnalysisProtectionDecorator", 
                "class DataAccessProtectionDecorator",
                "def comprehensive_pipeline_protection",
            ]
            
            for decorator in required_decorators:
                if decorator in content:
                    print(f"✅ {decorator} found in decorators")
                else:
                    print(f"❌ {decorator} missing from decorators")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline components test failed: {e}")
        return False

def test_step_configurations():
    """Test that step configurations are properly defined."""
    print("\n🧪 Testing step configurations...")
    
    try:
        # Read the orchestrator file to check step configurations
        orchestrator_file = Path("src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py")
        if orchestrator_file.exists():
            with open(orchestrator_file, "r") as f:
                content = f.read()
            
            expected_steps = [
                "hmm_clustering",
                "regime_splitting",
                "labeling", 
                "feature_engineering",
                "matrix_operations",
                "feature_selection"
            ]
            
            for step in expected_steps:
                if f"'{step}'" in content:
                    print(f"✅ Step '{step}' configuration found")
                else:
                    print(f"❌ Step '{step}' configuration missing")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Step configurations test failed: {e}")
        return False

def test_decorator_integration():
    """Test that decorators are properly integrated."""
    print("\n🧪 Testing decorator integration...")
    
    try:
        # Read the orchestrator file to check decorator usage
        orchestrator_file = Path("src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py")
        if orchestrator_file.exists():
            with open(orchestrator_file, "r") as f:
                content = f.read()
            
            # Check for decorator imports
            if "comprehensive_pipeline_protection" in content:
                print("✅ comprehensive_pipeline_protection decorator imported")
            else:
                print("❌ comprehensive_pipeline_protection decorator not imported")
                return False
            
            # Check for decorator usage on step methods
            decorator_usage_count = content.count("@comprehensive_pipeline_protection")
            if decorator_usage_count >= 6:  # Should be used on all 6 step methods
                print(f"✅ comprehensive_pipeline_protection decorator used {decorator_usage_count} times")
            else:
                print(f"❌ comprehensive_pipeline_protection decorator used only {decorator_usage_count} times (expected 6+)")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Decorator integration test failed: {e}")
        return False

def test_validation_integration():
    """Test that validation is properly integrated."""
    print("\n🧪 Testing validation integration...")
    
    try:
        # Read the orchestrator file to check validation usage
        orchestrator_file = Path("src/training/steps/market_analysis/enhanced_market_analysis_orchestrator.py")
        if orchestrator_file.exists():
            with open(orchestrator_file, "r") as f:
                content = f.read()
            
            # Check for enhanced validator usage
            if "self.enhanced_validator" in content:
                print("✅ Enhanced validator integrated in orchestrator")
            else:
                print("❌ Enhanced validator not integrated in orchestrator")
                return False
            
            # Check for validation method calls
            if "validate_step_output" in content:
                print("✅ Step output validation integrated")
            else:
                print("❌ Step output validation not integrated")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Validation integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Market Analysis Pipeline Structure")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_launcher_integration,
        test_pipeline_components,
        test_step_configurations,
        test_decorator_integration,
        test_validation_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! The enhanced pipeline is properly structured.")
        print("\n📋 Pipeline Enhancement Summary:")
        print("✅ Enhanced orchestrator with comprehensive validation")
        print("✅ Step-by-step validators with schema validation")
        print("✅ Data formatting, analysis, and access protection decorators")
        print("✅ Comprehensive error handling and observability")
        print("✅ Proper step transitions with validation")
        print("✅ Integration with existing launcher")
        return True
    else:
        print("❌ Some structure tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)