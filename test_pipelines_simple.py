#!/usr/bin/env python3
"""
Simple Three Pipelines Comparison Test

Compares the three pipeline implementations with a simpler approach.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class SimplePipelinesComparisonTester:
    """Simple tester for comparing the three pipeline implementations."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up a simple test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Test environment: {self.temp_dir}")
        
        # Create simple test files
        test_files = {
            "syntax_error.py": '''
def broken_function(
    # Missing closing parenthesis
    return "broken"
''',
            "import_issues.py": '''
import os
import sys
import os  # Duplicate import

def main():
    print("Hello world")
''',
            "clean_file.py": '''
import os

def clean_function():
    return "clean"
'''
        }
        
        for filename, content in test_files.items():
            test_file = self.temp_dir / filename
            test_file.write_text(content)
        
        return str(self.temp_dir)
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_imports(self) -> Dict[str, Any]:
        """Test that all three pipelines can be imported."""
        print("\n=== Testing Pipeline Imports ===")
        
        results = {
            "sequential_import": False,
            "enhanced_import": False,
            "standalone_import": False,
            "all_imports_work": False
        }
        
        try:
            # Test Sequential Fixer import
            try:
                from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
                results["sequential_import"] = True
                print("✓ Sequential Fixer Pipeline import successful")
            except Exception as e:
                print(f"✗ Sequential Fixer Pipeline import failed: {e}")
            
            # Test Unified Enhanced import
            try:
                from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
                results["enhanced_import"] = True
                print("✓ Unified Enhanced Pipeline import successful")
            except Exception as e:
                print(f"✗ Unified Enhanced Pipeline import failed: {e}")
            
            # Test Unified Standalone import
            try:
                from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
                results["standalone_import"] = True
                print("✓ Unified Standalone Pipeline import successful")
            except Exception as e:
                print(f"✗ Unified Standalone Pipeline import failed: {e}")
            
            # Check if all imports work
            if all([results["sequential_import"], results["enhanced_import"], results["standalone_import"]]):
                results["all_imports_work"] = True
                print("✓ All pipeline imports successful")
            
        except Exception as e:
            print(f"✗ Pipeline import test failed: {e}")
            
        return results
    
    def test_pipeline_creation(self) -> Dict[str, Any]:
        """Test that all three pipelines can be created."""
        print("\n=== Testing Pipeline Creation ===")
        
        results = {
            "sequential_creation": False,
            "enhanced_creation": False,
            "standalone_creation": False,
            "all_creations_work": False
        }
        
        try:
            # Test Sequential Fixer creation
            try:
                from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
                pipeline = SequentialFixer()
                results["sequential_creation"] = True
                print("✓ Sequential Fixer Pipeline creation successful")
            except Exception as e:
                print(f"✗ Sequential Fixer Pipeline creation failed: {e}")
            
            # Test Unified Enhanced creation
            try:
                from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
                pipeline = UnifiedEnhancedPipeline()
                results["enhanced_creation"] = True
                print("✓ Unified Enhanced Pipeline creation successful")
            except Exception as e:
                print(f"✗ Unified Enhanced Pipeline creation failed: {e}")
            
            # Test Unified Standalone creation
            try:
                from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
                pipeline = UnifiedStandalonePipeline()
                results["standalone_creation"] = True
                print("✓ Unified Standalone Pipeline creation successful")
            except Exception as e:
                print(f"✗ Unified Standalone Pipeline creation failed: {e}")
            
            # Check if all creations work
            if all([results["sequential_creation"], results["enhanced_creation"], results["standalone_creation"]]):
                results["all_creations_work"] = True
                print("✓ All pipeline creations successful")
            
        except Exception as e:
            print(f"✗ Pipeline creation test failed: {e}")
            
        return results
    
    def test_pipeline_execution(self) -> Dict[str, Any]:
        """Test that all three pipelines can execute."""
        print("\n=== Testing Pipeline Execution ===")
        
        results = {
            "sequential_execution": False,
            "enhanced_execution": False,
            "standalone_execution": False,
            "all_executions_work": False,
            "execution_times": {},
            "execution_results": {}
        }
        
        test_dir = self.setup_test_environment()
        try:
            # Test Sequential Fixer execution
            try:
                from code_quality.fixers.sequential_fixer_fixed import SequentialFixer
                pipeline = SequentialFixer()
                
                start_time = time.time()
                result = pipeline.run_pipeline(
                    target=str(test_dir),
                    output_dir=str(test_dir / "output"),
                    create_backups=True,
                    run_pre_commit=False
                )
                execution_time = time.time() - start_time
                
                results["execution_times"]["sequential"] = execution_time
                results["execution_results"]["sequential"] = result
                results["sequential_execution"] = True
                print(f"✓ Sequential Fixer Pipeline execution successful in {execution_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Sequential Fixer Pipeline execution failed: {e}")
            
            # Test Unified Enhanced execution
            try:
                from code_quality.pipelines.pipeline_unified_enhanced_fixed import UnifiedEnhancedPipeline
                pipeline = UnifiedEnhancedPipeline(project_root=str(test_dir))
                
                start_time = time.time()
                result = pipeline.run_all()
                execution_time = time.time() - start_time
                
                results["execution_times"]["enhanced"] = execution_time
                results["execution_results"]["enhanced"] = result
                results["enhanced_execution"] = True
                print(f"✓ Unified Enhanced Pipeline execution successful in {execution_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Unified Enhanced Pipeline execution failed: {e}")
            
            # Test Unified Standalone execution
            try:
                from code_quality.pipelines.pipeline_unified_standalone_fixed import UnifiedStandalonePipeline
                pipeline = UnifiedStandalonePipeline(project_root=str(test_dir))
                
                start_time = time.time()
                result = pipeline.run_all()
                execution_time = time.time() - start_time
                
                results["execution_times"]["standalone"] = execution_time
                results["execution_results"]["standalone"] = result
                results["standalone_execution"] = True
                print(f"✓ Unified Standalone Pipeline execution successful in {execution_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Unified Standalone Pipeline execution failed: {e}")
            
            # Check if all executions work
            if all([results["sequential_execution"], results["enhanced_execution"], results["standalone_execution"]]):
                results["all_executions_work"] = True
                print("✓ All pipeline executions successful")
            
        except Exception as e:
            print(f"✗ Pipeline execution test failed: {e}")
        finally:
            self.cleanup_test_environment()
            
        return results
    
    def compare_pipeline_results(self) -> Dict[str, Any]:
        """Compare the results from all three pipelines."""
        print("\n=== Comparing Pipeline Results ===")
        
        results = {
            "performance_comparison": {},
            "feature_comparison": {},
            "recommendations": {}
        }
        
        try:
            execution_results = self.test_results.get("pipeline_execution", {})
            execution_times = execution_results.get("execution_times", {})
            pipeline_results = execution_results.get("execution_results", {})
            
            # Performance comparison
            if execution_times:
                fastest_pipeline = min(execution_times.items(), key=lambda x: x[1])
                results["performance_comparison"] = {
                    "fastest": fastest_pipeline[0],
                    "fastest_time": fastest_pipeline[1],
                    "all_times": execution_times,
                    "speed_ranking": sorted(execution_times.items(), key=lambda x: x[1])
                }
                print(f"✓ Performance comparison completed")
                print(f"  Fastest pipeline: {fastest_pipeline[0]} ({fastest_pipeline[1]:.2f}s)")
            
            # Feature comparison
            features = {
                "sequential": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "plugin_system": False,
                    "parallel_execution": False,
                    "subprocess_execution": False,
                    "comprehensive_reporting": False
                },
                "enhanced": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "plugin_system": True,
                    "parallel_execution": True,
                    "subprocess_execution": False,
                    "comprehensive_reporting": True
                },
                "standalone": {
                    "syntax_fixing": True,
                    "import_fixing": True,
                    "linter_analysis": True,
                    "plugin_system": False,
                    "parallel_execution": True,
                    "subprocess_execution": True,
                    "comprehensive_reporting": True
                }
            }
            
            # Calculate feature scores
            feature_scores = {}
            for pipeline, pipeline_features in features.items():
                score = sum(1 for has_feature in pipeline_features.values() if has_feature)
                feature_scores[pipeline] = {
                    "score": score,
                    "total": len(pipeline_features),
                    "percentage": (score / len(pipeline_features)) * 100
                }
            
            results["feature_comparison"] = {
                "feature_matrix": features,
                "feature_scores": feature_scores,
                "best_featured": max(feature_scores.items(), key=lambda x: x[1]['score'])
            }
            
            print(f"✓ Feature comparison completed")
            for pipeline, score_info in feature_scores.items():
                print(f"  {pipeline}: {score_info['score']}/{score_info['total']} features ({score_info['percentage']:.1f}%)")
            
            # Generate recommendations
            recommendations = {
                "fastest": results["performance_comparison"].get("fastest", "unknown"),
                "most_featured": results["feature_comparison"].get("best_featured", ("unknown", {}))[0],
                "use_cases": {
                    "simple_projects": "sequential",
                    "complex_projects": "enhanced", 
                    "ci_cd_integration": "standalone"
                }
            }
            
            results["recommendations"] = recommendations
            print(f"✓ Recommendations generated")
            
        except Exception as e:
            print(f"✗ Pipeline results comparison failed: {e}")
            
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline comparison tests."""
        print("="*80)
        print("SIMPLE THREE PIPELINES COMPARISON TESTING")
        print("="*80)
        
        self.test_results = {
            "pipeline_imports": self.test_pipeline_imports(),
            "pipeline_creation": self.test_pipeline_creation(),
            "pipeline_execution": self.test_pipeline_execution()
        }
        
        # Compare results
        self.test_results["results_comparison"] = self.compare_pipeline_results()
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive comparison summary."""
        print("\n" + "="*80)
        print("THREE PIPELINES COMPARISON SUMMARY")
        print("="*80)
        
        # Import results
        import_results = self.test_results.get("pipeline_imports", {})
        print(f"\n📦 IMPORT RESULTS:")
        print(f"  Sequential Fixer: {'✓' if import_results.get('sequential_import') else '✗'}")
        print(f"  Unified Enhanced: {'✓' if import_results.get('enhanced_import') else '✗'}")
        print(f"  Unified Standalone: {'✓' if import_results.get('standalone_import') else '✗'}")
        print(f"  All Imports: {'✓' if import_results.get('all_imports_work') else '✗'}")
        
        # Creation results
        creation_results = self.test_results.get("pipeline_creation", {})
        print(f"\n🏗️ CREATION RESULTS:")
        print(f"  Sequential Fixer: {'✓' if creation_results.get('sequential_creation') else '✗'}")
        print(f"  Unified Enhanced: {'✓' if creation_results.get('enhanced_creation') else '✗'}")
        print(f"  Unified Standalone: {'✓' if creation_results.get('standalone_creation') else '✗'}")
        print(f"  All Creations: {'✓' if creation_results.get('all_creations_work') else '✗'}")
        
        # Execution results
        execution_results = self.test_results.get("pipeline_execution", {})
        print(f"\n🚀 EXECUTION RESULTS:")
        print(f"  Sequential Fixer: {'✓' if execution_results.get('sequential_execution') else '✗'}")
        print(f"  Unified Enhanced: {'✓' if execution_results.get('enhanced_execution') else '✗'}")
        print(f"  Unified Standalone: {'✓' if execution_results.get('standalone_execution') else '✗'}")
        print(f"  All Executions: {'✓' if execution_results.get('all_executions_work') else '✗'}")
        
        # Performance comparison
        results_comparison = self.test_results.get("results_comparison", {})
        perf_comparison = results_comparison.get("performance_comparison", {})
        if perf_comparison:
            print(f"\n⚡ PERFORMANCE COMPARISON:")
            print(f"  Fastest Pipeline: {perf_comparison.get('fastest', 'Unknown')}")
            print(f"  Execution Times:")
            for pipeline, time_taken in perf_comparison.get('all_times', {}).items():
                print(f"    {pipeline}: {time_taken:.2f}s")
        
        # Feature comparison
        feature_comparison = results_comparison.get("feature_comparison", {})
        if feature_comparison:
            print(f"\n🔧 FEATURE COMPARISON:")
            feature_scores = feature_comparison.get("feature_scores", {})
            for pipeline, score_info in feature_scores.items():
                print(f"  {pipeline}: {score_info['score']}/{score_info['total']} features ({score_info['percentage']:.1f}%)")
            
            best_featured = feature_comparison.get("best_featured", ("unknown", {}))
            print(f"  Most Featured: {best_featured[0]}")
        
        # Recommendations
        recommendations = results_comparison.get("recommendations", {})
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"  🏃 For Speed: {recommendations.get('fastest', 'Unknown')} Pipeline")
            print(f"  🔧 For Features: {recommendations.get('most_featured', 'Unknown')} Pipeline")
            print(f"\n  📋 Use Case Recommendations:")
            use_cases = recommendations.get("use_cases", {})
            print(f"    • Simple Projects: {use_cases.get('simple_projects', 'Unknown')} Pipeline")
            print(f"    • Complex Projects: {use_cases.get('complex_projects', 'Unknown')} Pipeline")
            print(f"    • CI/CD Integration: {use_cases.get('ci_cd_integration', 'Unknown')} Pipeline")
        
        # Overall assessment
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in self.test_results.items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    if isinstance(result, bool):
                        total_tests += 1
                        if result:
                            passed_tests += 1
        
        overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"  Test Score: {overall_score:.1f}% ({passed_tests}/{total_tests})")
        
        if overall_score >= 80:
            print(f"  Status: ✅ EXCELLENT - All pipelines working well")
        elif overall_score >= 60:
            print(f"  Status: ✅ GOOD - Most pipelines working")
        elif overall_score >= 40:
            print(f"  Status: ⚠️ FAIR - Some pipelines have issues")
        else:
            print(f"  Status: ❌ NEEDS WORK - Multiple pipeline issues")
        
        # Save results
        report_path = "/workspace/simple_pipelines_comparison_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")
        
        return overall_score


def main():
    """Main test runner."""
    tester = SimplePipelinesComparisonTester()
    
    try:
        results = tester.run_all_tests()
        overall_score = tester.print_summary()
        
        if overall_score >= 80:
            print("\n✅ Three pipelines comparison completed successfully!")
            return 0
        elif overall_score >= 60:
            print("\n⚠️ Three pipelines comparison completed with some issues")
            return 1
        else:
            print("\n❌ Three pipelines comparison found significant issues")
            return 2
            
    except Exception as e:
        print(f"Test suite failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())