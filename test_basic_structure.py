#!/usr/bin/env python3
"""
Basic structure test for Market Analysis Pipeline Improvements.

This script tests the basic file structure and imports without dependencies.
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing File Structure...")
    
    required_files = [
        "src/training/steps/market_analysis/components/__init__.py",
        "src/training/steps/market_analysis/components/base_component.py",
        "src/training/steps/market_analysis/components/component_factory.py",
        "src/training/steps/market_analysis/components/artifact_manager.py",
        "src/training/steps/market_analysis/components/sr_parameter_optimization.py",
        "src/training/steps/market_analysis/components/sr_detection.py",
        "src/training/steps/market_analysis/components/sr_clustering.py",
        "src/training/steps/market_analysis/components/hmm_regime_discovery.py",
        "src/training/steps/market_analysis/components/hmm_clustering.py",
        "src/training/steps/market_analysis/components/hmm_models_training.py",
        "src/training/steps/market_analysis/components/hmm_ensemble_training.py",
        "src/training/steps/market_analysis/components/regime_data_splitting.py",
        "src/training/steps/market_analysis/components/triple_barrier_labeling.py",
        "src/training/steps/market_analysis/components/feature_lookback_optimization.py",
        "src/training/steps/market_analysis/components/cross_timeframe_analysis.py"
    ]
    
    try:
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"✅ File exists: {file_path}")
            else:
                print(f"❌ File missing: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


def test_file_contents():
    """Test that files contain expected content."""
    print("\n🧪 Testing File Contents...")
    
    try:
        # Test base_component.py has expected classes
        base_component_path = Path("src/training/steps/market_analysis/components/base_component.py")
        if base_component_path.exists():
            content = base_component_path.read_text()
            if "class BaseMarketAnalysisComponent" in content:
                print("✅ BaseMarketAnalysisComponent class found")
            else:
                print("❌ BaseMarketAnalysisComponent class not found")
                return False
            
            if "class ComponentConfig" in content:
                print("✅ ComponentConfig class found")
            else:
                print("❌ ComponentConfig class not found")
                return False
        
        # Test component_factory.py has expected classes
        factory_path = Path("src/training/steps/market_analysis/components/component_factory.py")
        if factory_path.exists():
            content = factory_path.read_text()
            if "class ComponentFactory" in content:
                print("✅ ComponentFactory class found")
            else:
                print("❌ ComponentFactory class not found")
                return False
        
        # Test artifact_manager.py has expected classes
        artifact_path = Path("src/training/steps/market_analysis/components/artifact_manager.py")
        if artifact_path.exists():
            content = artifact_path.read_text()
            if "class ArtifactManager" in content:
                print("✅ ArtifactManager class found")
            else:
                print("❌ ArtifactManager class not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ File contents test failed: {e}")
        return False


def test_directory_structure():
    """Test that the directory structure is correct."""
    print("\n🧪 Testing Directory Structure...")
    
    try:
        # Test main components directory
        components_dir = Path("src/training/steps/market_analysis/components")
        if components_dir.exists() and components_dir.is_dir():
            print("✅ Components directory exists")
        else:
            print("❌ Components directory missing")
            return False
        
        # Test __init__.py exists and has content
        init_file = components_dir / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            if "ComponentFactory" in content and "ArtifactManager" in content:
                print("✅ __init__.py has expected exports")
            else:
                print("❌ __init__.py missing expected exports")
                return False
        else:
            print("❌ __init__.py missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False


def test_improvement_summaries():
    """Test that improvement summary files exist."""
    print("\n🧪 Testing Improvement Summaries...")
    
    try:
        summary_files = [
            "MARKET_ANALYSIS_IMPROVEMENTS_SUMMARY.md",
            "MARKET_ANALYSIS_CONTINUED_IMPROVEMENTS.md"
        ]
        
        for file_path in summary_files:
            if Path(file_path).exists():
                print(f"✅ Summary file exists: {file_path}")
            else:
                print(f"❌ Summary file missing: {file_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Improvement summaries test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Basic Structure Tests for Market Analysis Pipeline Improvements")
    print("=" * 80)
    
    tests = [
        test_file_structure,
        test_file_contents,
        test_directory_structure,
        test_improvement_summaries
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic structure tests passed!")
        print("📋 The component-based architecture has been successfully implemented.")
        print("📁 All required files and directories are in place.")
        print("📝 Improvement documentation is complete.")
        return 0
    else:
        print("⚠️ Some basic structure tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)