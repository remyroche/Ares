#!/usr/bin/env python3
"""
Detailed Pipeline Functionality Test

Tests all three pipelines with comprehensive output and verification.
"""

import json
import sys
import tempfile
import time
import os
from pathlib import Path
from typing import Any, Dict

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class PipelineFunctionalityTester:
    """Detailed tester for all three pipeline implementations."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up a comprehensive test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Test environment: {self.temp_dir}")
        
        # Create test Python files with various issues
        test_files = {
            "syntax_error.py": '''
def broken_function(
    # Missing closing parenthesis
    return "broken"

def another_broken():
    if True
        print("missing colon")

def yet_another():
    x = 1
    y = 2
    z = x + y
    return z
''',
            "import_issues.py": '''
import os
import sys
import os  # Duplicate import
from typing import List, Dict
from typing import Optional  # Duplicate from import
import json
import json  # Another duplicate

def unused_function():
    return "unused"

def main():
    print("Hello world")
    # Using os but not sys, json, or typing
    print(os.getcwd())
''',
            "style_issues.py": '''
import os,sys,json  # Multiple imports on one line
from typing import List, Dict, Optional

def poorly_formatted_function(  x: int,  y: str,  z: List[str]  ) -> str:
    """Poorly formatted function with bad spacing."""
    result = ""
    for item in z:
        result += item + " "
    return result.strip()

class BadlyFormattedClass:
    def __init__(self):
        self.x=1
        self.y=2
        self.z=3
    
    def method(self):
        if self.x>0 and self.y>0:
            return True
        else:
            return False
''',
            "clean_file.py": '''
import os
from typing import List

def clean_function(items: List[str]) -> str:
    """A clean function with proper syntax and formatting."""
    return " ".join(items)

if __name__ == "__main__":
    result = clean_function(["hello", "world"])
    print(result)
'''
        }
        
        # Write test files
        for filename, content in test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        return str(self.temp_dir)
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_sequential_pipeline_detailed(self) -> Dict[str, Any]:
        """Test Sequential Fixer Pipeline with detailed output."""
        print("\n" + "="*60)
        print("TESTING SEQUENTIAL FIXER PIPELINE")
        print("="*60)
        
        results = {
            "import_success": False,
            "creation_success": False,
            "execution_success": False,
            "execution_time": 0.0,
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "output_files_created": [],
            "error_messages": [],
            "detailed_results": {}
        }
        
        try:
            # Test import
            print("1. Testing import...")
            try:
                from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
                results["import_success"] = True
                print("   ✓ Import successful")
            except Exception as e:
                results["error_messages"].append(f"Import failed: {e}")
                print(f"   ✗ Import failed: {e}")
                return results
            
            # Test creation
            print("2. Testing pipeline creation...")
            try:
                pipeline = SequentialFixer()
                results["creation_success"] = True
                print("   ✓ Pipeline created successfully")
                print(f"   - Project root: {pipeline.project_root}")
                print(f"   - Config type: {type(pipeline.config)}")
            except Exception as e:
                results["error_messages"].append(f"Creation failed: {e}")
                print(f"   ✗ Creation failed: {e}")
                return results
            
            # Test execution
            print("3. Testing pipeline execution...")
            test_dir = self.setup_test_environment()
            try:
                start_time = time.time()
                
                result = pipeline.run_pipeline(
                    target=str(test_dir),
                    output_dir=str(test_dir / "output"),
                    create_backups=True,
                    run_pre_commit=False
                )
                
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
                
                print(f"   ✓ Execution completed in {execution_time:.2f} seconds")
                
                # Analyze results
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["files_processed"] = result.get("total_files_processed", 0)
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["detailed_results"] = result
                    
                    print(f"   - Files processed: {results['files_processed']}")
                    print(f"   - Issues found: {results['issues_found']}")
                    print(f"   - Issues fixed: {results['issues_fixed']}")
                    
                    # Check for output files
                    output_dir = test_dir / "output"
                    if output_dir.exists():
                        output_files = list(output_dir.rglob("*"))
                        results["output_files_created"] = [str(f) for f in output_files]
                        print(f"   - Output files created: {len(output_files)}")
                    
                    # Print detailed results
                    print("   - Detailed results:")
                    for key, value in result.items():
                        if isinstance(value, (dict, list)) and len(str(value)) > 100:
                            print(f"     {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"     {key}: {value}")
                else:
                    results["error_messages"].append(f"Invalid result type: {type(result)}")
                    print(f"   ✗ Invalid result type: {type(result)}")
                
            except Exception as e:
                results["error_messages"].append(f"Execution failed: {e}")
                print(f"   ✗ Execution failed: {e}")
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            results["error_messages"].append(f"Unexpected error: {e}")
            print(f"✗ Unexpected error: {e}")
        
        return results
    
    def test_enhanced_pipeline_detailed(self) -> Dict[str, Any]:
        """Test Unified Enhanced Pipeline with detailed output."""
        print("\n" + "="*60)
        print("TESTING UNIFIED ENHANCED PIPELINE")
        print("="*60)
        
        results = {
            "import_success": False,
            "creation_success": False,
            "execution_success": False,
            "execution_time": 0.0,
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "plugins_discovered": 0,
            "output_files_created": [],
            "error_messages": [],
            "detailed_results": {}
        }
        
        try:
            # Test import
            print("1. Testing import...")
            try:
                from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
                results["import_success"] = True
                print("   ✓ Import successful")
            except Exception as e:
                results["error_messages"].append(f"Import failed: {e}")
                print(f"   ✗ Import failed: {e}")
                return results
            
            # Test creation
            print("2. Testing pipeline creation...")
            try:
                test_dir = self.setup_test_environment()
                pipeline = UnifiedEnhancedPipeline(project_root=str(test_dir))
                results["creation_success"] = True
                print("   ✓ Pipeline created successfully")
                print(f"   - Project root: {pipeline.project_root}")
                print(f"   - Reports dir: {pipeline.reports_dir}")
                
                # Check for plugin discovery
                if hasattr(pipeline, 'plugin_registry'):
                    available_plugins = pipeline.plugin_registry.get_available_plugins()
                    results["plugins_discovered"] = len(available_plugins)
                    print(f"   - Plugins discovered: {len(available_plugins)}")
                    print(f"   - Available plugins: {available_plugins}")
                
            except Exception as e:
                results["error_messages"].append(f"Creation failed: {e}")
                print(f"   ✗ Creation failed: {e}")
                return results
            
            # Test execution
            print("3. Testing pipeline execution...")
            try:
                start_time = time.time()
                
                result = pipeline.run_all()
                
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
                
                print(f"   ✓ Execution completed in {execution_time:.2f} seconds")
                
                # Analyze results
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["files_processed"] = result.get("total_files_processed", 0)
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["detailed_results"] = result
                    
                    print(f"   - Files processed: {results['files_processed']}")
                    print(f"   - Issues found: {results['issues_found']}")
                    print(f"   - Issues fixed: {results['issues_fixed']}")
                    
                    # Check for output files
                    if pipeline.reports_dir.exists():
                        output_files = list(pipeline.reports_dir.rglob("*"))
                        results["output_files_created"] = [str(f) for f in output_files]
                        print(f"   - Output files created: {len(output_files)}")
                    
                    # Print detailed results
                    print("   - Detailed results:")
                    for key, value in result.items():
                        if isinstance(value, (dict, list)) and len(str(value)) > 100:
                            print(f"     {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"     {key}: {value}")
                else:
                    results["error_messages"].append(f"Invalid result type: {type(result)}")
                    print(f"   ✗ Invalid result type: {type(result)}")
                
            except Exception as e:
                results["error_messages"].append(f"Execution failed: {e}")
                print(f"   ✗ Execution failed: {e}")
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            results["error_messages"].append(f"Unexpected error: {e}")
            print(f"✗ Unexpected error: {e}")
        
        return results
    
    def test_standalone_pipeline_detailed(self) -> Dict[str, Any]:
        """Test Unified Standalone Pipeline with detailed output."""
        print("\n" + "="*60)
        print("TESTING UNIFIED STANDALONE PIPELINE")
        print("="*60)
        
        results = {
            "import_success": False,
            "creation_success": False,
            "execution_success": False,
            "execution_time": 0.0,
            "files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "tools_available": 0,
            "output_files_created": [],
            "error_messages": [],
            "detailed_results": {}
        }
        
        try:
            # Test import
            print("1. Testing import...")
            try:
                from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
                results["import_success"] = True
                print("   ✓ Import successful")
            except Exception as e:
                results["error_messages"].append(f"Import failed: {e}")
                print(f"   ✗ Import failed: {e}")
                return results
            
            # Test creation
            print("2. Testing pipeline creation...")
            try:
                test_dir = self.setup_test_environment()
                pipeline = UnifiedStandalonePipeline(project_root=str(test_dir))
                results["creation_success"] = True
                print("   ✓ Pipeline created successfully")
                print(f"   - Project root: {pipeline.project_root}")
                print(f"   - Reports dir: {pipeline.reports_dir}")
                
                # Check for available tools
                if hasattr(pipeline, 'tools'):
                    results["tools_available"] = len(pipeline.tools)
                    print(f"   - Tools available: {len(pipeline.tools)}")
                    print(f"   - Tool categories: {list(pipeline.tools.keys())}")
                
            except Exception as e:
                results["error_messages"].append(f"Creation failed: {e}")
                print(f"   ✗ Creation failed: {e}")
                return results
            
            # Test execution
            print("3. Testing pipeline execution...")
            try:
                start_time = time.time()
                
                result = pipeline.run_all()
                
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
                
                print(f"   ✓ Execution completed in {execution_time:.2f} seconds")
                
                # Analyze results
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["files_processed"] = result.get("total_files_processed", 0)
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["detailed_results"] = result
                    
                    print(f"   - Files processed: {results['files_processed']}")
                    print(f"   - Issues found: {results['issues_found']}")
                    print(f"   - Issues fixed: {results['issues_fixed']}")
                    
                    # Check for output files
                    if pipeline.reports_dir.exists():
                        output_files = list(pipeline.reports_dir.rglob("*"))
                        results["output_files_created"] = [str(f) for f in output_files]
                        print(f"   - Output files created: {len(output_files)}")
                    
                    # Print detailed results
                    print("   - Detailed results:")
                    for key, value in result.items():
                        if isinstance(value, (dict, list)) and len(str(value)) > 100:
                            print(f"     {key}: {type(value).__name__} with {len(value)} items")
                        else:
                            print(f"     {key}: {value}")
                else:
                    results["error_messages"].append(f"Invalid result type: {type(result)}")
                    print(f"   ✗ Invalid result type: {type(result)}")
                
            except Exception as e:
                results["error_messages"].append(f"Execution failed: {e}")
                print(f"   ✗ Execution failed: {e}")
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            results["error_messages"].append(f"Unexpected error: {e}")
            print(f"✗ Unexpected error: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline functionality tests."""
        print("="*80)
        print("DETAILED PIPELINE FUNCTIONALITY TESTING")
        print("="*80)
        
        self.test_results = {
            "sequential_pipeline": self.test_sequential_pipeline_detailed(),
            "enhanced_pipeline": self.test_enhanced_pipeline_detailed(),
            "standalone_pipeline": self.test_standalone_pipeline_detailed()
        }
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive functionality summary."""
        print("\n" + "="*80)
        print("PIPELINE FUNCTIONALITY SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        for pipeline_name, results in self.test_results.items():
            print(f"\n{pipeline_name.replace('_', ' ').title()}:")
            
            # Count individual tests
            pipeline_tests = 0
            pipeline_passed = 0
            
            for test_name, result in results.items():
                if isinstance(result, bool):
                    pipeline_tests += 1
                    total_tests += 1
                    if result:
                        pipeline_passed += 1
                        passed_tests += 1
                        print(f"  ✓ {test_name}")
                    else:
                        print(f"  ✗ {test_name}")
            
            # Pipeline-specific metrics
            if results.get("execution_success"):
                print(f"  📊 Execution Time: {results.get('execution_time', 0):.2f}s")
                print(f"  📁 Files Processed: {results.get('files_processed', 0)}")
                print(f"  🔍 Issues Found: {results.get('issues_found', 0)}")
                print(f"  🔧 Issues Fixed: {results.get('issues_fixed', 0)}")
                print(f"  📄 Output Files: {len(results.get('output_files_created', []))}")
            
            # Special metrics
            if "plugins_discovered" in results:
                print(f"  🔌 Plugins Discovered: {results.get('plugins_discovered', 0)}")
            if "tools_available" in results:
                print(f"  🛠️ Tools Available: {results.get('tools_available', 0)}")
            
            # Error messages
            if results.get("error_messages"):
                print(f"  ⚠️ Errors: {len(results['error_messages'])}")
                for error in results["error_messages"][:3]:  # Show first 3 errors
                    print(f"    - {error}")
            
            # Pipeline score
            if pipeline_tests > 0:
                score = (pipeline_passed / pipeline_tests) * 100
                print(f"  Score: {score:.1f}% ({pipeline_passed}/{pipeline_tests})")
        
        # Overall assessment
        overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed Tests: {passed_tests}")
        print(f"  Overall Score: {overall_score:.1f}%")
        
        # Functional status
        functional_pipelines = 0
        for pipeline_name, results in self.test_results.items():
            if (results.get("import_success") and 
                results.get("creation_success") and 
                results.get("execution_success")):
                functional_pipelines += 1
        
        print(f"\n🎯 FUNCTIONAL STATUS:")
        print(f"  Functional Pipelines: {functional_pipelines}/3")
        
        if functional_pipelines == 3:
            print(f"  Status: ✅ ALL PIPELINES FUNCTIONAL")
        elif functional_pipelines >= 2:
            print(f"  Status: ✅ MOST PIPELINES FUNCTIONAL")
        elif functional_pipelines >= 1:
            print(f"  Status: ⚠️ SOME PIPELINES FUNCTIONAL")
        else:
            print(f"  Status: ❌ NO PIPELINES FUNCTIONAL")
        
        # Save results
        report_path = "/workspace/pipeline_functionality_test_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return overall_score, functional_pipelines


def main():
    """Main test runner."""
    tester = PipelineFunctionalityTester()
    
    try:
        results = tester.run_all_tests()
        overall_score, functional_pipelines = tester.print_summary()
        
        if functional_pipelines == 3:
            print("\n✅ All three pipelines are fully functional!")
            return 0
        elif functional_pipelines >= 2:
            print("\n⚠️ Most pipelines are functional with some issues")
            return 1
        elif functional_pipelines >= 1:
            print("\n⚠️ Some pipelines are functional, others have issues")
            return 2
        else:
            print("\n❌ No pipelines are functional")
            return 3
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 4


if __name__ == "__main__":
    sys.exit(main())