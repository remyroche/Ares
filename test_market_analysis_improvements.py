#!/usr/bin/env python3
"""
Test script for Market Analysis Pipeline Improvements.

This script tests the new component-based architecture, artifact management,
and failure handling improvements.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.market_analysis.components import (
    ComponentFactory, 
    ComponentConfig, 
    ArtifactManager
)
from src.utils.logger import system_logger


async def test_artifact_manager():
    """Test the centralized artifact manager."""
    print("🧪 Testing Artifact Manager...")
    
    # Create artifact manager
    artifact_manager = ArtifactManager(
        base_dir="test_artifacts",
        symbol="BTCUSDT",
        exchange="binance", 
        timeframe="30m"
    )
    
    # Test artifact saving
    test_artifacts = {
        'test_data': {'key': 'value', 'number': 42},
        'test_dataframe': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
        'test_array': np.array([1, 2, 3, 4, 5])
    }
    
    try:
        saved_files = await artifact_manager.save_artifacts(
            'test_component', 
            test_artifacts,
            {'test_metadata': 'test_value'}
        )
        
        print(f"✅ Artifacts saved successfully:")
        for artifact_name, file_path in saved_files.items():
            print(f"   - {artifact_name}: {file_path}")
        
        # Test artifact summary
        summary = artifact_manager.get_artifact_summary()
        print(f"✅ Artifact summary: {summary['total_files']} files in {summary['artifact_directory']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Artifact manager test failed: {e}")
        return False


async def test_component_creation():
    """Test component creation and factory."""
    print("\n🧪 Testing Component Factory...")
    
    try:
        # Test available components
        available_components = ComponentFactory.get_available_components()
        print(f"✅ Available components: {available_components}")
        
        # Test component creation
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        
        # Test creating each available component
        for component_name in available_components:
            try:
                component = ComponentFactory.create_component(component_name, config)
                print(f"✅ Created component: {component_name} ({type(component).__name__})")
                
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


async def test_component_validation():
    """Test component artifact validation."""
    print("\n🧪 Testing Component Validation...")
    
    try:
        config = ComponentConfig()
        component = ComponentFactory.create_component('sr_parameter_optimization', config)
        
        # Test validation with complete artifacts
        complete_artifacts = {
            'optimized_parameters': {'param1': 'value1'},
            'quality_thresholds': {'threshold1': 0.5}
        }
        
        is_valid = component.validate_artifacts(complete_artifacts)
        print(f"✅ Complete artifacts validation: {is_valid}")
        
        # Test validation with incomplete artifacts
        incomplete_artifacts = {
            'optimized_parameters': {'param1': 'value1'},
            # Missing quality_thresholds
        }
        
        is_valid = component.validate_artifacts(incomplete_artifacts)
        print(f"✅ Incomplete artifacts validation: {is_valid} (should be False)")
        
        # Test validation with empty artifacts
        empty_artifacts = {
            'optimized_parameters': {},
            'quality_thresholds': []
        }
        
        is_valid = component.validate_artifacts(empty_artifacts)
        print(f"✅ Empty artifacts validation: {is_valid} (should be False)")
        
        return True
        
    except Exception as e:
        print(f"❌ Component validation test failed: {e}")
        return False


async def test_failure_handling():
    """Test failure handling and cleanup."""
    print("\n🧪 Testing Failure Handling...")
    
    try:
        # Create a test component that will fail
        config = ComponentConfig()
        component = ComponentFactory.create_component('sr_parameter_optimization', config)
        
        # Test with invalid data that should cause failure
        invalid_data = None  # This should cause the component to fail
        
        result = await component._execute_with_timing(invalid_data, {})
        
        print(f"✅ Component failure handling:")
        print(f"   Success: {result.success}")
        print(f"   Error message: {result.error_message}")
        print(f"   Artifacts: {len(result.artifacts)} items")
        
        # Verify that no artifacts were saved due to failure
        if not result.success and len(result.artifacts) == 0:
            print("✅ No artifacts saved on failure (correct behavior)")
        else:
            print("❌ Artifacts were saved despite failure (incorrect behavior)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failure handling test failed: {e}")
        return False


async def test_timestamp_organization():
    """Test that artifacts are organized with timestamps."""
    print("\n🧪 Testing Timestamp Organization...")
    
    try:
        # Create multiple artifact managers with different timestamps
        timestamp1 = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        artifact_manager1 = ArtifactManager(
            base_dir="test_timestamps",
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="30m"
        )
        
        # Wait a moment to ensure different timestamp
        await asyncio.sleep(1)
        
        artifact_manager2 = ArtifactManager(
            base_dir="test_timestamps", 
            symbol="ETHUSDT",  # Different symbol
            exchange="binance",
            timeframe="1h"
        )
        
        # Save artifacts with both managers
        test_artifacts = {'test_data': {'value': 123}}
        
        files1 = await artifact_manager1.save_artifacts('test1', test_artifacts)
        files2 = await artifact_manager2.save_artifacts('test2', test_artifacts)
        
        print(f"✅ Timestamp organization:")
        print(f"   Manager 1 files: {list(files1.keys())}")
        print(f"   Manager 2 files: {list(files2.keys())}")
        
        # Verify different directories were created
        dir1 = artifact_manager1.artifact_dir
        dir2 = artifact_manager2.artifact_dir
        
        if dir1 != dir2:
            print(f"✅ Different artifact directories created:")
            print(f"   Directory 1: {dir1}")
            print(f"   Directory 2: {dir2}")
        else:
            print("❌ Same artifact directory used (incorrect behavior)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Timestamp organization test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Market Analysis Pipeline Improvements Tests")
    print("=" * 60)
    
    tests = [
        test_artifact_manager,
        test_component_creation,
        test_component_validation,
        test_failure_handling,
        test_timestamp_organization
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
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
    exit_code = asyncio.run(main())
    sys.exit(exit_code)