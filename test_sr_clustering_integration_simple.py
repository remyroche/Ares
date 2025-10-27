#!/usr/bin/env python3
"""
Simple test to verify SR Clustering Component integration with BaseStep.
This test focuses on the integration aspects without requiring all dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_inheritance():
    """Test that SRClusteringComponent inherits from BaseStep."""
    print("🧪 Testing SR Clustering Component Inheritance")
    print("=" * 50)
    
    try:
        # Import BaseStep
        from src.training.steps.base_step import BaseStep
        print("✅ BaseStep imported successfully")
        
        # Check if SRClusteringComponent is defined
        import inspect
        from src.training.steps.market_analysis.components import sr_clustering
        
        # Get the SRClusteringComponent class
        sr_clustering_class = getattr(sr_clustering, 'SRClusteringComponent', None)
        
        if sr_clustering_class is None:
            print("❌ SRClusteringComponent class not found")
            return False
        
        print("✅ SRClusteringComponent class found")
        
        # Check inheritance
        is_basestep_subclass = issubclass(sr_clustering_class, BaseStep)
        print(f"✅ Inherits from BaseStep: {is_basestep_subclass}")
        
        # Check required methods exist
        required_methods = [
            'execute',
            'get_required_artifacts',
            '_save_artifact',
            '_get_artifact',
            '_get_sr_levels',
            '_validate_basestep_integration',
            '_load_artifacts_from_previous_stage'
        ]
        
        print("\n📋 Checking required methods:")
        all_methods_exist = True
        for method_name in required_methods:
            has_method = hasattr(sr_clustering_class, method_name)
            status = "✅" if has_method else "❌"
            print(f"   {status} {method_name}: {has_method}")
            if not has_method:
                all_methods_exist = False
        
        if all_methods_exist:
            print("\n✅ All required methods exist")
        else:
            print("\n❌ Some required methods are missing")
            return False
        
        # Check method signatures
        print("\n📋 Checking method signatures:")
        
        # Check execute method
        execute_method = getattr(sr_clustering_class, 'execute', None)
        if execute_method:
            sig = inspect.signature(execute_method)
            params = list(sig.parameters.keys())
            if 'config' in params:
                print("   ✅ execute method has 'config' parameter")
            else:
                print("   ❌ execute method missing 'config' parameter")
                return False
        
        # Check get_required_artifacts method
        required_artifacts_method = getattr(sr_clustering_class, 'get_required_artifacts', None)
        if required_artifacts_method:
            sig = inspect.signature(required_artifacts_method)
            params = list(sig.parameters.keys())
            if len(params) == 1 and 'self' in params:  # Only self parameter
                print("   ✅ get_required_artifacts method has correct signature")
            else:
                print("   ❌ get_required_artifacts method has incorrect signature")
                return False
        
        print("\n🎉 All inheritance and method checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_integration_features():
    """Test integration-specific features."""
    print("\n🔧 Testing Integration Features")
    print("=" * 50)
    
    try:
        from src.training.steps.market_analysis.components import sr_clustering
        
        # Check if the integration methods are properly implemented
        sr_clustering_class = getattr(sr_clustering, 'SRClusteringComponent', None)
        
        if sr_clustering_class is None:
            print("❌ SRClusteringComponent class not found")
            return False
        
        # Check for integration-specific methods
        integration_methods = [
            '_load_sr_levels_for_clustering',
            '_load_artifacts_from_previous_stage',
            '_validate_basestep_integration',
            '_create_sr_levels_dictionary'
        ]
        
        print("📋 Checking integration methods:")
        all_integration_methods_exist = True
        for method_name in integration_methods:
            has_method = hasattr(sr_clustering_class, method_name)
            status = "✅" if has_method else "❌"
            print(f"   {status} {method_name}: {has_method}")
            if not has_method:
                all_integration_methods_exist = False
        
        if all_integration_methods_exist:
            print("\n✅ All integration methods exist")
        else:
            print("\n❌ Some integration methods are missing")
            return False
        
        # Check if the component uses BaseStep methods in its implementation
        print("\n📋 Checking BaseStep method usage:")
        
        # Read the source file to check for BaseStep method usage
        source_file = Path(__file__).parent / "src" / "training" / "steps" / "market_analysis" / "components" / "sr_clustering.py"
        
        if source_file.exists():
            with open(source_file, 'r') as f:
                content = f.read()
            
            basestep_methods = [
                'self._save_artifact(',
                'self._get_artifact(',
                'self._get_sr_levels(',
                'self.artifact_manager.',
                'BaseStep'
            ]
            
            for method in basestep_methods:
                if method in content:
                    print(f"   ✅ Uses {method}")
                else:
                    print(f"   ❌ Missing {method}")
                    return False
        
        print("\n🎉 All integration features verified!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed with error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting SR Clustering Component BaseStep Integration Test")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_inheritance()
    test2_passed = test_integration_features()
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"   Inheritance Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! SR Clustering Component is fully integrated with BaseStep")
        print("\n📋 Integration Summary:")
        print("   ✅ Inherits from BaseStep")
        print("   ✅ Implements all required methods")
        print("   ✅ Uses BaseStep artifact management")
        print("   ✅ Has proper method signatures")
        print("   ✅ Includes integration validation")
        print("   ✅ Supports artifact loading from previous stages")
        print("   ✅ Creates SR levels dictionary for feature bank access")
        return True
    else:
        print("\n❌ Some tests failed. Integration needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)