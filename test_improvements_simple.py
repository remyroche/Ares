#!/usr/bin/env python3
"""
Simple test script for Market Analysis Pipeline Improvements.

This script tests the basic functionality without external dependencies.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_artifact_manager_basic():
    """Test basic artifact manager functionality."""
    print("🧪 Testing Artifact Manager (Basic)...")
    
    try:
        from src.training.steps.market_analysis.components.artifact_manager import ArtifactManager
        
        # Create artifact manager
        artifact_manager = ArtifactManager(
            base_dir="test_artifacts",
            symbol="BTCUSDT",
            exchange="binance", 
            timeframe="30m"
        )
        
        print(f"✅ Artifact manager created")
        print(f"   Directory: {artifact_manager.artifact_dir}")
        print(f"   Session timestamp: {artifact_manager.session_timestamp}")
        
        # Test artifact path generation
        path = artifact_manager.get_artifact_path('test_component', 'test_artifact', 'json')
        print(f"✅ Artifact path generated: {path}")
        
        # Test JSON serialization
        test_data = {
            'string': 'test',
            'number': 42,
            'list': [1, 2, 3],
            'dict': {'key': 'value'},
            'timestamp': datetime.now()
        }
        
        serialized = artifact_manager._json_serializer(test_data)
        print(f"✅ JSON serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Artifact manager test failed: {e}")
        return False


def test_component_factory():
    """Test component factory functionality."""
    print("\n🧪 Testing Component Factory...")
    
    try:
        from src.training.steps.market_analysis.components.component_factory import ComponentFactory
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        # Test available components
        available_components = ComponentFactory.get_available_components()
        print(f"✅ Available components: {available_components}")
        
        # Test component availability check
        for component_name in available_components:
            is_available = ComponentFactory.is_component_available(component_name)
            print(f"✅ Component {component_name} available: {is_available}")
        
        # Test component creation (without execution)
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        
        for component_name in available_components:
            try:
                component = ComponentFactory.create_component(component_name, config)
                print(f"✅ Created component: {component_name}")
                
                # Test required artifacts
                required_artifacts = component.get_required_artifacts()
                print(f"   Required artifacts: {required_artifacts}")
                
            except Exception as e:
                print(f"❌ Failed to create component {component_name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component factory test failed: {e}")
        return False


def test_base_component():
    """Test base component functionality."""
    print("\n🧪 Testing Base Component...")
    
    try:
        from src.training.steps.market_analysis.components.base_component import ComponentConfig, ComponentResult
        
        # Test ComponentConfig
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m",
            custom_params={'test': 'value'}
        )
        
        print(f"✅ ComponentConfig created: {config.symbol}")
        
        # Test ComponentResult
        result = ComponentResult(
            success=True,
            artifacts={'test_artifact': 'test_value'},
            metadata={'test_meta': 'test_meta_value'}
        )
        
        print(f"✅ ComponentResult created: success={result.success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Base component test failed: {e}")
        return False


def test_imports():
    """Test that all imports work correctly."""
    print("\n🧪 Testing Imports...")
    
    try:
        # Test main component imports
        from src.training.steps.market_analysis.components import (
            ComponentFactory,
            ComponentConfig,
            ArtifactManager
        )
        print("✅ Main component imports successful")
        
        # Test individual component imports
        from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationComponent
        from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
        from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
        from src.training.steps.market_analysis.components.hmm_regime_discovery import HMMRegimeDiscoveryComponent
        
        print("✅ Individual component imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        "src/training/steps/market_analysis/components/__init__.py",
        "src/training/steps/market_analysis/components/base_component.py",
        "src/training/steps/market_analysis/components/component_factory.py",
        "src/training/steps/market_analysis/components/artifact_manager.py",
        "src/training/steps/market_analysis/components/sr_parameter_optimization.py",
        "src/training/steps/market_analysis/components/sr_detection.py",
        "src/training/steps/market_analysis/components/sr_clustering.py",
        "src/training/steps/market_analysis/components/hmm_regime_discovery.py"
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


def main():
    """Run all tests."""
    print("🚀 Starting Market Analysis Pipeline Improvements Tests (Simple)")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_file_structure,
        test_base_component,
        test_component_factory,
        test_artifact_manager_basic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improvements are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)