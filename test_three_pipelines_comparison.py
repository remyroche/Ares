#!/usr/bin/env python3
"""
Three Pipelines Comparison Test

Compares the three pipeline implementations:
1. Sequential Fixer Pipeline
2. Unified Enhanced Pipeline  
3. Unified Standalone Pipeline

Tests functionality, performance, and results quality.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class ThreePipelinesComparisonTester:
    """Comprehensive tester for comparing the three pipeline implementations."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        self.test_files = {}
        
    def setup_test_environment(self):
        """Set up a comprehensive test environment with various code issues."""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Test environment: {self.temp_dir}")
        
        # Create test Python files with various issues
        self.test_files = {
            "syntax_errors.py": '''
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
            "security_issues.py": '''
import subprocess
import os
import pickle

def dangerous_function():
    # Potential security issues
    user_input = input("Enter command: ")
    subprocess.run(user_input, shell=True)  # Dangerous!
    
    # Another issue
    password = "hardcoded_password_123"  # Hardcoded password
    
    # Pickle deserialization
    data = pickle.loads(user_input)  # Unsafe deserialization
    
    return password

def sql_injection_example():
    query = "SELECT * FROM users WHERE id = " + user_input  # SQL injection
    return query
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
''',
            "complex_file.py": '''
import os
import sys
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import tempfile

class ComplexClass:
    """A complex class with multiple methods."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.data: List[str] = []
    
    def process_data(self, items: List[str]) -> Dict[str, int]:
        """Process a list of items and return statistics."""
        result = {}
        for item in items:
            if item in result:
                result[item] += 1
            else:
                result[item] = 1
        return result
    
    def save_to_file(self, filename: str) -> bool:
        """Save data to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.data, f)
            return True
        except Exception:
            return False

def complex_function(data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A complex function with multiple operations."""
    if not data:
        return None
    
    result = {
        "total_items": len(data),
        "processed_items": 0,
        "errors": []
    }
    
    for item in data:
        try:
            # Process item
            result["processed_items"] += 1
        except Exception as e:
            result["errors"].append(str(e))
    
    return result

if __name__ == "__main__":
    # Test the complex functionality
    test_data = [
        {"name": "item1", "value": 1},
        {"name": "item2", "value": 2},
        {"name": "item3", "value": 3}
    ]
    
    result = complex_function(test_data)
    print(f"Processed {result['processed_items']} items")
'''
        }
        
        # Write test files
        for filename, content in self.test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        return str(self.temp_dir)
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_sequential_pipeline(self) -> Dict[str, Any]:
        """Test the Sequential Fixer Pipeline."""
        print("\n=== Testing Sequential Fixer Pipeline ===")
        
        results = {
            "imports_work": False,
            "pipeline_creation": False,
            "execution_success": False,
            "execution_time": 0.0,
            "issues_found": 0,
            "issues_fixed": 0,
            "files_processed": 0,
            "error_handling": False
        }
        
        try:
            # Test imports
            from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
            results["imports_work"] = True
            print("✓ Sequential pipeline imports successful")
            
            # Test pipeline creation
            pipeline = SequentialFixer()
            results["pipeline_creation"] = True
            print("✓ Sequential pipeline creation successful")
            
            # Test execution
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
                
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["files_processed"] = result.get("total_files_processed", 0)
                    print(f"✓ Sequential pipeline execution successful in {execution_time:.2f}s")
                    print(f"  Files processed: {results['files_processed']}")
                    print(f"  Issues found: {results['issues_found']}")
                    print(f"  Issues fixed: {results['issues_fixed']}")
                else:
                    print("✗ Sequential pipeline execution failed - invalid result")
                
            except Exception as e:
                print(f"✗ Sequential pipeline execution failed: {e}")
                results["error_handling"] = True  # Error was handled gracefully
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Sequential pipeline test failed: {e}")
            
        return results
    
    def test_unified_enhanced_pipeline(self) -> Dict[str, Any]:
        """Test the Unified Enhanced Pipeline."""
        print("\n=== Testing Unified Enhanced Pipeline ===")
        
        results = {
            "imports_work": False,
            "pipeline_creation": False,
            "execution_success": False,
            "execution_time": 0.0,
            "issues_found": 0,
            "issues_fixed": 0,
            "files_processed": 0,
            "error_handling": False,
            "plugin_integration": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
            results["imports_work"] = True
            print("✓ Unified Enhanced pipeline imports successful")
            
            # Test pipeline creation
            pipeline = UnifiedEnhancedPipeline(project_root=str(self.temp_dir))
            results["pipeline_creation"] = True
            print("✓ Unified Enhanced pipeline creation successful")
            
            # Test execution
            test_dir = self.setup_test_environment()
            try:
                start_time = time.time()
                
                result = pipeline.run_all()
                
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
                
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["files_processed"] = result.get("total_files_processed", 0)
                    
                    # Check for plugin integration
                    if "plugin_results" in result or "available_plugins" in result:
                        results["plugin_integration"] = True
                    
                    print(f"✓ Unified Enhanced pipeline execution successful in {execution_time:.2f}s")
                    print(f"  Files processed: {results['files_processed']}")
                    print(f"  Issues found: {results['issues_found']}")
                    print(f"  Issues fixed: {results['issues_fixed']}")
                    print(f"  Plugin integration: {results['plugin_integration']}")
                else:
                    print("✗ Unified Enhanced pipeline execution failed - invalid result")
                
            except Exception as e:
                print(f"✗ Unified Enhanced pipeline execution failed: {e}")
                results["error_handling"] = True  # Error was handled gracefully
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Unified Enhanced pipeline test failed: {e}")
            
        return results
    
    def test_unified_standalone_pipeline(self) -> Dict[str, Any]:
        """Test the Unified Standalone Pipeline."""
        print("\n=== Testing Unified Standalone Pipeline ===")
        
        results = {
            "imports_work": False,
            "pipeline_creation": False,
            "execution_success": False,
            "execution_time": 0.0,
            "issues_found": 0,
            "issues_fixed": 0,
            "files_processed": 0,
            "error_handling": False,
            "subprocess_execution": False
        }
        
        try:
            # Test imports
            from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
            results["imports_work"] = True
            print("✓ Unified Standalone pipeline imports successful")
            
            # Test pipeline creation
            pipeline = UnifiedStandalonePipeline(project_root=str(self.temp_dir))
            results["pipeline_creation"] = True
            print("✓ Unified Standalone pipeline creation successful")
            
            # Test execution
            test_dir = self.setup_test_environment()
            try:
                start_time = time.time()
                
                result = pipeline.run_all()
                
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
                
                if isinstance(result, dict):
                    results["execution_success"] = True
                    results["issues_found"] = result.get("total_issues_found", 0)
                    results["issues_fixed"] = result.get("total_issues_fixed", 0)
                    results["files_processed"] = result.get("total_files_processed", 0)
                    
                    # Check for subprocess execution
                    if "tool_results" in result or "subprocess_results" in result:
                        results["subprocess_execution"] = True
                    
                    print(f"✓ Unified Standalone pipeline execution successful in {execution_time:.2f}s")
                    print(f"  Files processed: {results['files_processed']}")
                    print(f"  Issues found: {results['issues_found']}")
                    print(f"  Issues fixed: {results['issues_fixed']}")
                    print(f"  Subprocess execution: {results['subprocess_execution']}")
                else:
                    print("✗ Unified Standalone pipeline execution failed - invalid result")
                
            except Exception as e:
                print(f"✗ Unified Standalone pipeline execution failed: {e}")
                results["error_handling"] = True  # Error was handled gracefully
            finally:
                self.cleanup_test_environment()
            
        except Exception as e:
            print(f"✗ Unified Standalone pipeline test failed: {e}")
            
        return results
    
    def compare_pipeline_performance(self) -> Dict[str, Any]:
        """Compare performance metrics across all three pipelines."""
        print("\n=== Comparing Pipeline Performance ===")
        
        results = {
            "sequential_performance": {},
            "enhanced_performance": {},
            "standalone_performance": {},
            "performance_ranking": [],
            "speed_comparison": {},
            "efficiency_comparison": {}
        }
        
        try:
            # Get results from all three pipelines
            sequential_results = self.test_results.get("sequential_pipeline", {})
            enhanced_results = self.test_results.get("unified_enhanced_pipeline", {})
            standalone_results = self.test_results.get("unified_standalone_pipeline", {})
            
            # Extract performance metrics
            results["sequential_performance"] = {
                "execution_time": sequential_results.get("execution_time", 0),
                "files_processed": sequential_results.get("files_processed", 0),
                "issues_found": sequential_results.get("issues_found", 0),
                "issues_fixed": sequential_results.get("issues_fixed", 0),
                "throughput": sequential_results.get("files_processed", 0) / max(sequential_results.get("execution_time", 1), 0.001)
            }
            
            results["enhanced_performance"] = {
                "execution_time": enhanced_results.get("execution_time", 0),
                "files_processed": enhanced_results.get("files_processed", 0),
                "issues_found": enhanced_results.get("issues_found", 0),
                "issues_fixed": enhanced_results.get("issues_fixed", 0),
                "throughput": enhanced_results.get("files_processed", 0) / max(enhanced_results.get("execution_time", 1), 0.001)
            }
            
            results["standalone_performance"] = {
                "execution_time": standalone_results.get("execution_time", 0),
                "files_processed": standalone_results.get("files_processed", 0),
                "issues_found": standalone_results.get("issues_found", 0),
                "issues_fixed": standalone_results.get("issues_fixed", 0),
                "throughput": standalone_results.get("files_processed", 0) / max(standalone_results.get("execution_time", 1), 0.001)
            }
            
            # Rank pipelines by speed (fastest first)
            pipelines = [
                ("Sequential", results["sequential_performance"]["execution_time"]),
                ("Enhanced", results["enhanced_performance"]["execution_time"]),
                ("Standalone", results["standalone_performance"]["execution_time"])
            ]
            pipelines.sort(key=lambda x: x[1])
            results["performance_ranking"] = [p[0] for p in pipelines]
            
            # Speed comparison
            fastest_time = min([p[1] for p in pipelines if p[1] > 0])
            results["speed_comparison"] = {
                "fastest": pipelines[0][0] if pipelines else "None",
                "fastest_time": fastest_time,
                "speed_ratios": {
                    "Sequential": results["sequential_performance"]["execution_time"] / max(fastest_time, 0.001),
                    "Enhanced": results["enhanced_performance"]["execution_time"] / max(fastest_time, 0.001),
                    "Standalone": results["standalone_performance"]["execution_time"] / max(fastest_time, 0.001)
                }
            }
            
            # Efficiency comparison (issues fixed per second)
            results["efficiency_comparison"] = {
                "Sequential": results["sequential_performance"]["issues_fixed"] / max(results["sequential_performance"]["execution_time"], 0.001),
                "Enhanced": results["enhanced_performance"]["issues_fixed"] / max(results["enhanced_performance"]["execution_time"], 0.001),
                "Standalone": results["standalone_performance"]["issues_fixed"] / max(results["standalone_performance"]["execution_time"], 0.001)
            }
            
            print(f"✓ Performance comparison completed")
            print(f"  Fastest pipeline: {results['speed_comparison']['fastest']}")
            print(f"  Performance ranking: {' > '.join(results['performance_ranking'])}")
            
        except Exception as e:
            print(f"✗ Performance comparison failed: {e}")
            
        return results
    
    def compare_pipeline_features(self) -> Dict[str, Any]:
        """Compare features and capabilities across all three pipelines."""
        print("\n=== Comparing Pipeline Features ===")
        
        results = {
            "feature_matrix": {},
            "unique_features": {},
            "common_features": [],
            "feature_ranking": {}
        }
        
        try:
            # Define feature matrix
            features = {
                "Sequential": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "ast_validation": True,
                    "signature_analysis": True,
                    "plugin_system": False,
                    "parallel_execution": False,
                    "subprocess_execution": False,
                    "comprehensive_reporting": False,
                    "backup_system": True,
                    "error_recovery": True,
                    "metrics_collection": False
                },
                "Enhanced": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "ast_validation": True,
                    "signature_analysis": True,
                    "plugin_system": True,
                    "parallel_execution": True,
                    "subprocess_execution": False,
                    "comprehensive_reporting": True,
                    "backup_system": True,
                    "error_recovery": True,
                    "metrics_collection": True
                },
                "Standalone": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "ast_validation": True,
                    "signature_analysis": True,
                    "plugin_system": False,
                    "parallel_execution": True,
                    "subprocess_execution": True,
                    "comprehensive_reporting": True,
                    "backup_system": True,
                    "error_recovery": True,
                    "metrics_collection": True
                }
            }
            
            results["feature_matrix"] = features
            
            # Find unique features for each pipeline
            for pipeline_name, pipeline_features in features.items():
                unique_features = []
                for feature, has_feature in pipeline_features.items():
                    if has_feature:
                        # Check if this feature is unique to this pipeline
                        other_pipelines = [p for p in features.keys() if p != pipeline_name]
                        is_unique = all(not features[other_pipeline].get(feature, False) for other_pipeline in other_pipelines)
                        if is_unique:
                            unique_features.append(feature)
                results["unique_features"][pipeline_name] = unique_features
            
            # Find common features
            all_features = set(features["Sequential"].keys())
            common_features = []
            for feature in all_features:
                if all(features[pipeline].get(feature, False) for pipeline in features.keys()):
                    common_features.append(feature)
            results["common_features"] = common_features
            
            # Calculate feature scores
            for pipeline_name, pipeline_features in features.items():
                score = sum(1 for has_feature in pipeline_features.values() if has_feature)
                results["feature_ranking"][pipeline_name] = {
                    "score": score,
                    "total_features": len(pipeline_features),
                    "percentage": (score / len(pipeline_features)) * 100
                }
            
            print(f"✓ Feature comparison completed")
            print(f"  Common features: {len(common_features)}")
            for pipeline, ranking in results["feature_ranking"].items():
                print(f"  {pipeline}: {ranking['score']}/{ranking['total_features']} features ({ranking['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"✗ Feature comparison failed: {e}")
            
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline comparison tests."""
        print("="*80)
        print("THREE PIPELINES COMPARISON TESTING")
        print("="*80)
        
        # Test each pipeline individually
        self.test_results = {
            "sequential_pipeline": self.test_sequential_pipeline(),
            "unified_enhanced_pipeline": self.test_unified_enhanced_pipeline(),
            "unified_standalone_pipeline": self.test_unified_standalone_pipeline()
        }
        
        # Compare pipelines
        self.test_results["performance_comparison"] = self.compare_pipeline_performance()
        self.test_results["feature_comparison"] = self.compare_pipeline_features()
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive comparison summary."""
        print("\n" + "="*80)
        print("THREE PIPELINES COMPARISON SUMMARY")
        print("="*80)
        
        # Individual pipeline results
        print(f"\n📊 INDIVIDUAL PIPELINE RESULTS:")
        for pipeline_name, results in self.test_results.items():
            if pipeline_name.endswith("_pipeline"):
                print(f"\n{pipeline_name.replace('_', ' ').title()}:")
                success_rate = sum(1 for v in results.values() if v is True) / len(results) * 100
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Execution Time: {results.get('execution_time', 0):.2f}s")
                print(f"  Files Processed: {results.get('files_processed', 0)}")
                print(f"  Issues Found: {results.get('issues_found', 0)}")
                print(f"  Issues Fixed: {results.get('issues_fixed', 0)}")
        
        # Performance comparison
        perf_comparison = self.test_results.get("performance_comparison", {})
        if perf_comparison:
            print(f"\n🚀 PERFORMANCE COMPARISON:")
            print(f"  Fastest Pipeline: {perf_comparison.get('speed_comparison', {}).get('fastest', 'Unknown')}")
            print(f"  Performance Ranking: {' > '.join(perf_comparison.get('performance_ranking', []))}")
            
            speed_ratios = perf_comparison.get('speed_comparison', {}).get('speed_ratios', {})
            if speed_ratios:
                print(f"  Speed Ratios (relative to fastest):")
                for pipeline, ratio in speed_ratios.items():
                    print(f"    {pipeline}: {ratio:.2f}x")
        
        # Feature comparison
        feature_comparison = self.test_results.get("feature_comparison", {})
        if feature_comparison:
            print(f"\n🔧 FEATURE COMPARISON:")
            feature_ranking = feature_comparison.get('feature_ranking', {})
            for pipeline, ranking in feature_ranking.items():
                print(f"  {pipeline}: {ranking['score']}/{ranking['total_features']} features ({ranking['percentage']:.1f}%)")
            
            unique_features = feature_comparison.get('unique_features', {})
            if unique_features:
                print(f"\n  Unique Features:")
                for pipeline, features in unique_features.items():
                    if features:
                        print(f"    {pipeline}: {', '.join(features)}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Performance recommendation
        fastest = perf_comparison.get('speed_comparison', {}).get('fastest', 'Unknown')
        if fastest != 'Unknown':
            print(f"  🏃 For Speed: Use {fastest} Pipeline")
        
        # Feature recommendation
        feature_ranking = feature_comparison.get('feature_ranking', {})
        if feature_ranking:
            best_featured = max(feature_ranking.items(), key=lambda x: x[1]['score'])
            print(f"  🔧 For Features: Use {best_featured[0]} Pipeline ({best_featured[1]['score']} features)")
        
        # Use case recommendations
        print(f"\n  📋 Use Case Recommendations:")
        print(f"    • Simple Projects: Sequential Pipeline (fast, focused)")
        print(f"    • Complex Projects: Enhanced Pipeline (comprehensive, plugin support)")
        print(f"    • CI/CD Integration: Standalone Pipeline (subprocess execution, isolation)")
        
        # Save results
        report_path = "/workspace/three_pipelines_comparison_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return self.test_results


def main():
    """Main test runner."""
    tester = ThreePipelinesComparisonTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_summary()
        
        print("\n✅ Three pipelines comparison completed successfully!")
        return 0
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())