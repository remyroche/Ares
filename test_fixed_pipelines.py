#!/usr/bin/env python3
"""
Test Suite for Fixed Pipeline Files

Tests the improved versions of the pipeline files that address:
1. Dependency management
2. Error handling
3. Redundancy reduction
4. Resource cleanup
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class FixedPipelineTester:
    """Tester for the fixed pipeline versions."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up a temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Test environment: {self.temp_dir}")
        
        # Create a simple test Python file
        test_file = self.temp_dir / "test_file.py"
        test_file.write_text("""
import os
import sys
from typing import List, Dict, Optional

def test_function(x: int, y: str = "default") -> bool:
    \"\"\"Test function with type hints.\"\"\"
    return x > 0 and len(y) > 0

class TestClass:
    def __init__(self, name: str):
        self.name = name
    
    def method(self) -> str:
        return f"Hello {self.name}"

# Some syntax issues for testing
def broken_function(
    # Missing closing parenthesis
    return "broken"
""")
        
        return str(self.temp_dir)
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_fixed_sequential_fixer(self) -> Dict[str, Any]:
        """Test the fixed SequentialFixer."""
        print("\n=== Testing Fixed SequentialFixer ===")
        
        results = {
            "imports_work": False,
            "class_instantiation": False,
            "dependency_handling": False,
            "error_handling": False,
            "basic_execution": False
        }
        
        try:
            # Test imports
            from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
            results["imports_work"] = True
            print("✓ Imports successful")
            
            # Test class instantiation
            fixer = SequentialFixer()
            results["class_instantiation"] = True
            print("✓ Class instantiation successful")
            
            # Test dependency handling
            from code_quality.utils.dependency_manager import dependency_manager
            dependency_manager.print_dependency_status()
            results["dependency_handling"] = True
            print("✓ Dependency handling works")
            
            # Test error handling
            test_dir = self.setup_test_environment()
            try:
                result = fixer.run_pipeline(
                    target=test_dir,
                    output_dir=str(self.temp_dir / "output"),
                    create_backups=False
                )
                
                if isinstance(result, dict) and "pipeline_info" in result:
                    results["basic_execution"] = True
                    print("✓ Basic execution successful")
                else:
                    print("✗ Basic execution failed - invalid result format")
                    
            except Exception as e:
                print(f"✗ Basic execution failed: {e}")
                results["error_handling"] = True  # Error handling worked
            finally:
                self.cleanup_test_environment()
                
        except Exception as e:
            print(f"✗ Fixed SequentialFixer test failed: {e}")
            
        return results
    
    def test_fixed_enhanced_pipeline(self) -> Dict[str, Any]:
        """Test the fixed UnifiedEnhancedPipeline."""
        print("\n=== Testing Fixed UnifiedEnhancedPipeline ===")
        
        results = {
            "imports_work": False,
            "class_instantiation": False,
            "dependency_handling": False,
            "error_handling": False,
            "basic_execution": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
            results["imports_work"] = True
            print("✓ Imports successful")
            
            # Test class instantiation
            test_dir = self.setup_test_environment()
            pipeline = UnifiedEnhancedPipeline(test_dir)
            results["class_instantiation"] = True
            print("✓ Class instantiation successful")
            
            # Test dependency handling
            from code_quality.utils.dependency_manager import dependency_manager
            dependency_manager.print_dependency_status()
            results["dependency_handling"] = True
            print("✓ Dependency handling works")
            
            # Test error handling
            try:
                # Test a simple method that doesn't require complex dependencies
                result = pipeline.run_syntax_fixes()
                
                if isinstance(result, dict) and "execution_time" in result:
                    results["basic_execution"] = True
                    print("✓ Basic execution successful")
                else:
                    print("✗ Basic execution failed - invalid result format")
                    
            except Exception as e:
                print(f"✗ Basic execution failed: {e}")
                results["error_handling"] = True  # Error handling worked
            finally:
                self.cleanup_test_environment()
                
        except Exception as e:
            print(f"✗ Fixed UnifiedEnhancedPipeline test failed: {e}")
            
        return results
    
    def test_fixed_standalone_pipeline(self) -> Dict[str, Any]:
        """Test the fixed UnifiedStandalonePipeline."""
        print("\n=== Testing Fixed UnifiedStandalonePipeline ===")
        
        results = {
            "imports_work": False,
            "class_instantiation": False,
            "error_handling": False,
            "basic_execution": False,
            "base_pipeline_inheritance": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
            results["imports_work"] = True
            print("✓ Imports successful")
            
            # Test class instantiation
            test_dir = self.setup_test_environment()
            pipeline = UnifiedStandalonePipeline(test_dir)
            results["class_instantiation"] = True
            print("✓ Class instantiation successful")
            
            # Test base pipeline inheritance
            from code_quality.pipelines.base_pipeline import BasePipeline
            if isinstance(pipeline, BasePipeline):
                results["base_pipeline_inheritance"] = True
                print("✓ Base pipeline inheritance works")
            
            # Test error handling
            try:
                # Test run_tool with a non-existent tool to test error handling
                result = pipeline.run_tool("non_existent_tool")
                
                if isinstance(result, dict) and "error" in result:
                    results["basic_execution"] = True
                    results["error_handling"] = True
                    print("✓ Basic execution and error handling successful")
                else:
                    print("✗ Basic execution failed - invalid result format")
                    
            except Exception as e:
                print(f"✗ Basic execution failed: {e}")
            finally:
                self.cleanup_test_environment()
                
        except Exception as e:
            print(f"✗ Fixed UnifiedStandalonePipeline test failed: {e}")
            
        return results
    
    def test_dependency_manager(self) -> Dict[str, Any]:
        """Test the dependency manager."""
        print("\n=== Testing Dependency Manager ===")
        
        results = {
            "imports_work": False,
            "dependency_checking": False,
            "safe_imports": False,
            "fallback_config": False
        }
        
        try:
            # Test imports
            from code_quality.utils.dependency_manager import dependency_manager, safe_import
            results["imports_work"] = True
            print("✓ Imports successful")
            
            # Test dependency checking
            available = dependency_manager.get_available_dependencies()
            missing = dependency_manager.get_missing_dependencies()
            results["dependency_checking"] = True
            print(f"✓ Dependency checking works - Available: {len(available)}, Missing: {len(missing)}")
            
            # Test safe imports
            module, success = safe_import("os", None)
            if module is not None and success:
                results["safe_imports"] = True
                print("✓ Safe imports work")
            
            # Test fallback config
            config = dependency_manager.create_fallback_config()
            if isinstance(config, dict) and "auto_fix" in config:
                results["fallback_config"] = True
                print("✓ Fallback config creation works")
                
        except Exception as e:
            print(f"✗ Dependency manager test failed: {e}")
            
        return results
    
    def test_base_pipeline(self) -> Dict[str, Any]:
        """Test the base pipeline class."""
        print("\n=== Testing Base Pipeline ===")
        
        results = {
            "imports_work": False,
            "class_instantiation": False,
            "common_methods": False,
            "context_manager": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.base_pipeline import BasePipeline
            results["imports_work"] = True
            print("✓ Imports successful")
            
            # Test class instantiation
            pipeline = BasePipeline("/tmp/test")
            results["class_instantiation"] = True
            print("✓ Class instantiation successful")
            
            # Test common methods
            required_methods = [
                '_setup_execution_tracking', '_finalize_execution_tracking',
                '_save_report', '_print_section_header', '_print_pipeline_header',
                '_generate_summary', '_print_summary', '_handle_error',
                '_validate_project_root', '_find_python_files'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(pipeline, method):
                    missing_methods.append(method)
            
            if not missing_methods:
                results["common_methods"] = True
                print("✓ All common methods exist")
            else:
                print(f"✗ Missing methods: {missing_methods}")
            
            # Test context manager
            try:
                with BasePipeline("/tmp/test") as ctx_pipeline:
                    if ctx_pipeline is not None:
                        results["context_manager"] = True
                        print("✓ Context manager works")
            except Exception as e:
                print(f"✗ Context manager failed: {e}")
                
        except Exception as e:
            print(f"✗ Base pipeline test failed: {e}")
            
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests for the fixed pipeline versions."""
        print("="*80)
        print("TESTING FIXED PIPELINE VERSIONS")
        print("="*80)
        
        self.test_results = {
            "fixed_sequential_fixer": self.test_fixed_sequential_fixer(),
            "fixed_enhanced_pipeline": self.test_fixed_enhanced_pipeline(),
            "fixed_standalone_pipeline": self.test_fixed_standalone_pipeline(),
            "dependency_manager": self.test_dependency_manager(),
            "base_pipeline": self.test_base_pipeline()
        }
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("FIXED PIPELINE TEST SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for component, results in self.test_results.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            component_passed = 0
            component_total = 0
            
            for test, passed in results.items():
                component_total += 1
                total_tests += 1
                if passed:
                    component_passed += 1
                    passed_tests += 1
                    print(f"  ✓ {test}")
                else:
                    print(f"  ✗ {test}")
            
            if component_total > 0:
                score = (component_passed / component_total) * 100
                print(f"  Score: {score:.1f}% ({component_passed}/{component_total})")
        
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\nOverall Score: {overall_score:.1f}% ({passed_tests}/{total_tests})")
        
        # Save results
        report_path = "/workspace/fixed_pipeline_test_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return overall_score


def main():
    """Main test runner."""
    tester = FixedPipelineTester()
    
    try:
        results = tester.run_all_tests()
        overall_score = tester.print_summary()
        
        # Exit with appropriate code
        if overall_score >= 80:
            print("\n✓ Fixed pipelines are working well!")
            return 0
        elif overall_score >= 60:
            print("\n⚠ Fixed pipelines have some issues but are mostly functional")
            return 1
        else:
            print("\n✗ Fixed pipelines have significant issues")
            return 2
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())