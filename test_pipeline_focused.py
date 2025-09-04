#!/usr/bin/env python3
"""
Focused Test Suite for Pipeline Files - Addressing Specific Issues

This test focuses on the specific issues found in the comprehensive test:
1. Missing dependencies (rich, astroid)
2. Import resolution issues
3. Redundancy issues
4. Resource cleanup issues
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))


class FocusedPipelineTester:
    """Focused tester addressing specific issues."""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.test_results = {}
        
    def check_dependencies(self) -> Dict[str, Any]:
        """Check for missing dependencies and suggest fixes."""
        print("\n=== Checking Dependencies ===")
        
        results = {
            "missing_dependencies": [],
            "suggested_fixes": [],
            "dependency_issues": False
        }
        
        # Check for common missing dependencies
        missing_deps = []
        
        # Check if rich is available
        try:
            import rich
        except ImportError:
            missing_deps.append("rich")
        
        # Check if astroid is available
        try:
            import astroid
        except ImportError:
            missing_deps.append("astroid")
        
        # Check if other common dependencies are available
        common_deps = ["pylint", "flake8", "black", "isort", "mypy"]
        for dep in common_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        results["missing_dependencies"] = missing_deps
        
        if missing_deps:
            results["dependency_issues"] = True
            results["suggested_fixes"].append("Install missing dependencies: pip install " + " ".join(missing_deps))
            print(f"✗ Missing dependencies: {missing_deps}")
        else:
            print("✓ All dependencies available")
        
        return results
    
    def test_import_resolution(self) -> Dict[str, Any]:
        """Test import resolution with fallback strategies."""
        print("\n=== Testing Import Resolution ===")
        
        results = {
            "direct_imports": {},
            "fallback_imports": {},
            "import_issues": []
        }
        
        # Test direct imports
        import_tests = [
            ("code_quality.fixers.sequential_fixer", "SequentialFixer"),
            ("code_quality.pipelines.pipeline_unified_enhanced", "UnifiedEnhancedPipeline"),
            ("code_quality.pipelines.pipeline_unified_standalone", "UnifiedStandalonePipeline")
        ]
        
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                results["direct_imports"][f"{module_name}.{class_name}"] = True
                print(f"✓ {module_name}.{class_name} imports successfully")
            except Exception as e:
                results["direct_imports"][f"{module_name}.{class_name}"] = False
                results["import_issues"].append(f"{module_name}.{class_name}: {e}")
                print(f"✗ {module_name}.{class_name} import failed: {e}")
        
        # Test fallback imports (without optional dependencies)
        fallback_tests = [
            ("code_quality.core.config", "get_default_config"),
        ]
        
        for module_name, func_name in fallback_tests:
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                results["fallback_imports"][f"{module_name}.{func_name}"] = True
                print(f"✓ {module_name}.{func_name} fallback import successful")
            except Exception as e:
                results["fallback_imports"][f"{module_name}.{func_name}"] = False
                results["import_issues"].append(f"{module_name}.{func_name}: {e}")
                print(f"✗ {module_name}.{func_name} fallback import failed: {e}")
        
        return results
    
    def analyze_redundancy(self) -> Dict[str, Any]:
        """Analyze and suggest fixes for redundancy issues."""
        print("\n=== Analyzing Redundancy ===")
        
        results = {
            "duplicate_functions": [],
            "duplicate_imports": [],
            "suggested_consolidations": [],
            "redundancy_score": 0
        }
        
        files_to_analyze = [
            "/workspace/code_quality/fixers/sequential_fixer.py",
            "/workspace/code_quality/pipelines/pipeline_unified_enhanced.py",
            "/workspace/code_quality/pipelines/pipeline_unified_standalone.py"
        ]
        
        # Analyze function duplication
        function_analysis = {}
        for file_path in files_to_analyze:
            with open(file_path, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "file": file_path
                    })
            
            function_analysis[file_path] = functions
        
        # Find duplicate function names
        function_names = {}
        for file_path, functions in function_analysis.items():
            for func in functions:
                name = func["name"]
                if name not in function_names:
                    function_names[name] = []
                function_names[name].append(func)
        
        duplicates = {name: funcs for name, funcs in function_names.items() if len(funcs) > 1}
        results["duplicate_functions"] = duplicates
        
        # Analyze import duplication
        import_analysis = {}
        for file_path in files_to_analyze:
            with open(file_path, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            import_analysis[file_path] = imports
        
        # Find duplicate imports
        all_imports = set()
        for imports in import_analysis.values():
            all_imports.update(imports)
        
        import_counts = {}
        for import_name in all_imports:
            count = sum(1 for imports in import_analysis.values() if import_name in imports)
            if count > 1:
                import_counts[import_name] = count
        
        results["duplicate_imports"] = import_counts
        
        # Generate consolidation suggestions
        if duplicates:
            results["suggested_consolidations"].append("Consider creating a base class for common functionality")
            results["suggested_consolidations"].append("Move duplicate functions to a shared utility module")
        
        if import_counts:
            results["suggested_consolidations"].append("Create a common imports module to reduce duplication")
        
        # Calculate redundancy score
        total_issues = len(duplicates) + len(import_counts)
        if total_issues == 0:
            results["redundancy_score"] = 100
        else:
            results["redundancy_score"] = max(0, 100 - (total_issues * 10))
        
        print(f"Found {len(duplicates)} duplicate functions and {len(import_counts)} duplicate imports")
        print(f"Redundancy score: {results['redundancy_score']}/100")
        
        return results
    
    def test_resource_cleanup(self) -> Dict[str, Any]:
        """Test resource cleanup patterns."""
        print("\n=== Testing Resource Cleanup ===")
        
        results = {
            "cleanup_patterns": {},
            "missing_cleanup": [],
            "cleanup_score": 0
        }
        
        files_to_check = [
            "/workspace/code_quality/fixers/sequential_fixer.py",
            "/workspace/code_quality/pipelines/pipeline_unified_enhanced.py",
            "/workspace/code_quality/pipelines/pipeline_unified_standalone.py"
        ]
        
        cleanup_patterns = [
            "finally:", "with open(", "close()", "cleanup", "rmtree", "unlink",
            "tempfile", "contextmanager", "__enter__", "__exit__"
        ]
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            found_patterns = []
            for pattern in cleanup_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
            
            results["cleanup_patterns"][file_path] = found_patterns
            
            if not found_patterns:
                results["missing_cleanup"].append(file_path)
        
        # Calculate cleanup score
        total_files = len(files_to_check)
        files_with_cleanup = total_files - len(results["missing_cleanup"])
        results["cleanup_score"] = (files_with_cleanup / total_files) * 100
        
        print(f"Files with cleanup patterns: {files_with_cleanup}/{total_files}")
        print(f"Cleanup score: {results['cleanup_score']}/100")
        
        if results["missing_cleanup"]:
            print(f"Files missing cleanup: {results['missing_cleanup']}")
        
        return results
    
    def generate_fixes(self) -> Dict[str, Any]:
        """Generate specific fixes for identified issues."""
        print("\n=== Generating Fixes ===")
        
        fixes = {
            "dependency_fixes": [],
            "import_fixes": [],
            "redundancy_fixes": [],
            "cleanup_fixes": []
        }
        
        # Dependency fixes
        fixes["dependency_fixes"] = [
            "Add try/except blocks around optional imports",
            "Create fallback implementations for missing dependencies",
            "Add dependency checking in __init__ methods"
        ]
        
        # Import fixes
        fixes["import_fixes"] = [
            "Use relative imports consistently",
            "Add import error handling",
            "Create import fallback mechanisms"
        ]
        
        # Redundancy fixes
        fixes["redundancy_fixes"] = [
            "Create base Pipeline class with common functionality",
            "Move duplicate functions to utils module",
            "Consolidate common imports into shared module"
        ]
        
        # Cleanup fixes
        fixes["cleanup_fixes"] = [
            "Add finally blocks for resource cleanup",
            "Use context managers for file operations",
            "Add cleanup methods to classes"
        ]
        
        return fixes
    
    def run_focused_tests(self) -> Dict[str, Any]:
        """Run focused tests addressing specific issues."""
        print("="*80)
        print("FOCUSED PIPELINE TESTING")
        print("="*80)
        
        self.test_results = {
            "dependencies": self.check_dependencies(),
            "imports": self.test_import_resolution(),
            "redundancy": self.analyze_redundancy(),
            "cleanup": self.test_resource_cleanup(),
            "fixes": self.generate_fixes()
        }
        
        return self.test_results
    
    def print_focused_summary(self):
        """Print focused test summary with actionable recommendations."""
        print("\n" + "="*80)
        print("FOCUSED TEST SUMMARY")
        print("="*80)
        
        # Dependencies
        deps = self.test_results["dependencies"]
        print(f"Dependencies: {'✓' if not deps['dependency_issues'] else '✗'}")
        if deps["dependency_issues"]:
            print(f"  Missing: {deps['missing_dependencies']}")
            for fix in deps["suggested_fixes"]:
                print(f"  Fix: {fix}")
        
        # Imports
        imports = self.test_results["imports"]
        direct_success = sum(1 for success in imports["direct_imports"].values() if success)
        total_direct = len(imports["direct_imports"])
        print(f"Import Resolution: {direct_success}/{total_direct} successful")
        
        # Redundancy
        redundancy = self.test_results["redundancy"]
        print(f"Redundancy Score: {redundancy['redundancy_score']}/100")
        
        # Cleanup
        cleanup = self.test_results["cleanup"]
        print(f"Resource Cleanup Score: {cleanup['cleanup_score']}/100")
        
        # Overall recommendations
        print("\nPriority Fixes:")
        fixes = self.test_results["fixes"]
        
        if deps["dependency_issues"]:
            print("1. HIGH: Fix dependency issues")
            for fix in fixes["dependency_fixes"]:
                print(f"   - {fix}")
        
        if len(imports["import_issues"]) > 0:
            print("2. HIGH: Fix import resolution")
            for fix in fixes["import_fixes"]:
                print(f"   - {fix}")
        
        if redundancy["redundancy_score"] < 80:
            print("3. MEDIUM: Reduce redundancy")
            for fix in fixes["redundancy_fixes"]:
                print(f"   - {fix}")
        
        if cleanup["cleanup_score"] < 80:
            print("4. MEDIUM: Improve resource cleanup")
            for fix in fixes["cleanup_fixes"]:
                print(f"   - {fix}")
        
        # Save results
        report_path = "/workspace/focused_test_results.json"
        with open(report_path, "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\nDetailed results saved to: {report_path}")


def main():
    """Main focused test runner."""
    tester = FocusedPipelineTester()
    
    try:
        results = tester.run_focused_tests()
        tester.print_focused_summary()
        
        # Calculate overall health
        deps_ok = not results["dependencies"]["dependency_issues"]
        imports_ok = len(results["imports"]["import_issues"]) == 0
        redundancy_ok = results["redundancy"]["redundancy_score"] >= 70
        cleanup_ok = results["cleanup"]["cleanup_score"] >= 70
        
        if deps_ok and imports_ok and redundancy_ok and cleanup_ok:
            print("\n✓ All critical issues addressed!")
            return 0
        else:
            print("\n⚠ Some issues remain - see recommendations above")
            return 1
            
    except Exception as e:
        print(f"Focused test suite failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())