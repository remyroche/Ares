#!/usr/bin/env python3
"""
Comprehensive Import and Undefined Variable Checker

This script provides:
1. Required imports checking - ensures all necessary imports are present
2. Undefined variables detection - spots undefined variables for easier troubleshooting

Can be integrated into various pipelines for automated code quality checking.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add code_quality to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyzers.import_analyzer import ImportAnalyzer
    from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
    from core.config import get_default_config
except ImportError:
    # Fallback for when running as standalone script
    import importlib.util
    
    # Load modules dynamically
    def load_module_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # Load required modules
    import_analyzer_path = Path(__file__).parent / "analyzers" / "import_analyzer.py"
    undefined_analyzer_path = Path(__file__).parent / "analyzers" / "undefined_names_analyzer.py"
    config_path = Path(__file__).parent / "core" / "config.py"
    
    if import_analyzer_path.exists():
        import_analyzer_module = load_module_from_path("import_analyzer", import_analyzer_path)
        ImportAnalyzer = import_analyzer_module.ImportAnalyzer
    else:
        ImportAnalyzer = None
    
    if undefined_analyzer_path.exists():
        undefined_analyzer_module = load_module_from_path("undefined_names_analyzer", undefined_analyzer_path)
        UndefinedNamesAnalyzer = undefined_analyzer_module.UndefinedNamesAnalyzer
    else:
        UndefinedNamesAnalyzer = None
    
    if config_path.exists():
        config_module = load_module_from_path("config", config_path)
        get_default_config = config_module.get_default_config
    else:
        def get_default_config():
            return None


class ImportAndUndefinedChecker:
    """
    Comprehensive checker for imports and undefined variables.
    
    This class provides:
    1. Import validation - checks for missing, unused, conflicting imports
    2. Undefined variable detection - identifies undefined names and variables
    3. Report generation - creates detailed reports for troubleshooting
    """
    
    def __init__(self, project_root: str = None, config=None):
        """
        Initialize the checker.
        
        Args:
            project_root: Root directory of the project to analyze
            config: CodeQualityConfig instance (optional)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config = config or get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize analyzers
        if ImportAnalyzer is not None:
            self.import_analyzer = ImportAnalyzer(self.config)
        else:
            self.import_analyzer = None
            
        if UndefinedNamesAnalyzer is not None:
            self.undefined_analyzer = UndefinedNamesAnalyzer(self.config)
        else:
            self.undefined_analyzer = None
        
        # Results storage
        self.results = {
            "import_analysis": {},
            "undefined_analysis": {},
            "summary": {},
            "recommendations": []
        }
    
    def check_imports(self, target_path: str = None) -> Dict[str, Any]:
        """
        Check for import issues in the target path.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing import analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("🔍 Checking imports...")
        print(f"Target: {target_path}")
        
        start_time = time.time()
        
        if self.import_analyzer is None:
            return {
                "status": "error",
                "error": "ImportAnalyzer not available - check dependencies",
                "execution_time": 0,
                "target_path": target_path
            }
        
        try:
            if os.path.isfile(target_path):
                # Single file analysis
                results = self.import_analyzer.analyze_files([target_path])
            else:
                # Directory analysis
                results = self.import_analyzer.analyze_directory(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            import_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "issues": results.get("issues", {}),
                "files": results.get("files", {}),
                "import_graph": results.get("import_graph", {}),
            }
            
            # Count issues
            total_issues = 0
            issue_breakdown = {}
            
            for issue_type, issues in results.get("issues", {}).items():
                count = len(issues) if isinstance(issues, list) else 0
                issue_breakdown[issue_type] = count
                total_issues += count
            
            import_results["total_issues"] = total_issues
            import_results["issue_breakdown"] = issue_breakdown
            
            # Generate recommendations
            recommendations = []
            if issue_breakdown.get("duplicate_imports", 0) > 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "imports",
                    "message": f"Remove {issue_breakdown['duplicate_imports']} duplicate imports"
                })
            
            if issue_breakdown.get("circular_dependencies", 0) > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "imports",
                    "message": f"Resolve {issue_breakdown['circular_dependencies']} circular dependencies"
                })
            
            if issue_breakdown.get("unused_imports", 0) > 0:
                recommendations.append({
                    "priority": "low",
                    "category": "imports",
                    "message": f"Remove {issue_breakdown['unused_imports']} unused imports"
                })
            
            if issue_breakdown.get("conflicting_imports", 0) > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "imports",
                    "message": f"Resolve {issue_breakdown['conflicting_imports']} import conflicts"
                })
            
            if issue_breakdown.get("unresolvable_imports", 0) > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "imports",
                    "message": f"Fix {issue_breakdown['unresolvable_imports']} unresolvable imports"
                })
            
            import_results["recommendations"] = recommendations
            
            self.results["import_analysis"] = import_results
            
            # Print summary
            print(f"✅ Import analysis completed in {execution_time:.2f}s")
            print(f"📊 Total import issues: {total_issues}")
            for issue_type, count in issue_breakdown.items():
                if count > 0:
                    print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
            
            return import_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["import_analysis"] = error_result
            print(f"❌ Import analysis failed: {e}")
            return error_result
    
    def check_undefined_variables(self, target_path: str = None) -> Dict[str, Any]:
        """
        Check for undefined variables and names in the target path.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing undefined variable analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("🔍 Checking undefined variables...")
        print(f"Target: {target_path}")
        
        start_time = time.time()
        
        if self.undefined_analyzer is None:
            return {
                "status": "error",
                "error": "UndefinedNamesAnalyzer not available - check dependencies",
                "execution_time": 0,
                "target_path": target_path
            }
        
        try:
            if os.path.isfile(target_path):
                # Single file analysis
                results = self.undefined_analyzer.analyze_file(target_path)
            else:
                # Directory analysis
                results = self.undefined_analyzer.analyze_directory(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            undefined_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "files": results.get("files", {}),
            }
            
            # Extract summary information
            summary = results.get("summary", {})
            total_errors = summary.get("total_errors", 0)
            files_with_errors = summary.get("files_with_errors", 0)
            
            undefined_results["total_errors"] = total_errors
            undefined_results["files_with_errors"] = files_with_errors
            
            # Generate recommendations
            recommendations = []
            if total_errors > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "undefined_variables",
                    "message": f"Fix {total_errors} undefined variable/name issues"
                })
                
                # Add specific recommendations based on error types
                undefined_names = summary.get("undefined_names", 0)
                undefined_imports = summary.get("undefined_imports", 0)
                unused_imports = summary.get("unused_imports", 0)
                import_conflicts = summary.get("import_conflicts", 0)
                
                if undefined_names > 0:
                    recommendations.append({
                        "priority": "high",
                        "category": "undefined_variables",
                        "message": f"Define {undefined_names} undefined variables/functions"
                    })
                
                if undefined_imports > 0:
                    recommendations.append({
                        "priority": "high",
                        "category": "imports",
                        "message": f"Fix {undefined_imports} undefined import references"
                    })
                
                if unused_imports > 0:
                    recommendations.append({
                        "priority": "medium",
                        "category": "imports",
                        "message": f"Remove {unused_imports} unused imports"
                    })
                
                if import_conflicts > 0:
                    recommendations.append({
                        "priority": "high",
                        "category": "imports",
                        "message": f"Resolve {import_conflicts} import conflicts"
                    })
            
            undefined_results["recommendations"] = recommendations
            
            self.results["undefined_analysis"] = undefined_results
            
            # Print summary
            print(f"✅ Undefined variable analysis completed in {execution_time:.2f}s")
            print(f"📊 Total undefined issues: {total_errors}")
            print(f"📄 Files with issues: {files_with_errors}")
            
            if total_errors > 0:
                print("🔍 Issue breakdown:")
                for key, value in summary.items():
                    if key.startswith(("undefined_", "unused_", "import_")) and value > 0:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            return undefined_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["undefined_analysis"] = error_result
            print(f"❌ Undefined variable analysis failed: {e}")
            return error_result
    
    def run_comprehensive_check(self, target_path: str = None) -> Dict[str, Any]:
        """
        Run both import and undefined variable checks.
        
        Args:
            target_path: Path to analyze (default: project_root)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if target_path is None:
            target_path = str(self.project_root)
        
        print("="*70)
        print("COMPREHENSIVE IMPORT AND UNDEFINED VARIABLE CHECK")
        print("="*70)
        print(f"Target: {target_path}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        start_time = time.time()
        
        # Run both checks
        import_results = self.check_imports(target_path)
        print()
        undefined_results = self.check_undefined_variables(target_path)
        
        total_time = time.time() - start_time
        
        # Generate overall summary
        overall_summary = {
            "timestamp": self.timestamp,
            "target_path": target_path,
            "total_execution_time": total_time,
            "import_issues": import_results.get("total_issues", 0),
            "undefined_issues": undefined_results.get("total_errors", 0),
            "total_issues": (import_results.get("total_issues", 0) + 
                           undefined_results.get("total_errors", 0)),
            "files_with_import_issues": len(import_results.get("files", {})),
            "files_with_undefined_issues": undefined_results.get("files_with_errors", 0),
        }
        
        # Combine recommendations
        all_recommendations = []
        all_recommendations.extend(import_results.get("recommendations", []))
        all_recommendations.extend(undefined_results.get("recommendations", []))
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        overall_summary["recommendations"] = all_recommendations
        
        self.results["summary"] = overall_summary
        
        # Print final summary
        print("\n" + "="*70)
        print("COMPREHENSIVE CHECK SUMMARY")
        print("="*70)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Import issues: {overall_summary['import_issues']}")
        print(f"Undefined variable issues: {overall_summary['undefined_issues']}")
        print(f"Total issues: {overall_summary['total_issues']}")
        
        if all_recommendations:
            print("\n📋 Recommendations:")
            for i, rec in enumerate(all_recommendations, 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "low"), "⚪")
                print(f"  {i}. {priority_emoji} [{rec.get('priority', 'low').upper()}] {rec.get('message', '')}")
        
        return self.results
    
    def save_report(self, output_file: str = None) -> str:
        """
        Save the analysis results to a JSON report file.
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Path to the saved report file
        """
        if output_file is None:
            output_file = f"import_undefined_check_report_{self.timestamp}.json"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            raise
    
    def get_high_priority_issues(self) -> List[Dict[str, Any]]:
        """
        Get a list of high-priority issues that need immediate attention.
        
        Returns:
            List of high-priority issues
        """
        high_priority = []
        
        # Check import analysis
        import_analysis = self.results.get("import_analysis", {})
        if import_analysis.get("status") == "success":
            for rec in import_analysis.get("recommendations", []):
                if rec.get("priority") == "high":
                    high_priority.append({
                        "type": "import",
                        "category": rec.get("category"),
                        "message": rec.get("message"),
                        "source": "import_analysis"
                    })
        
        # Check undefined analysis
        undefined_analysis = self.results.get("undefined_analysis", {})
        if undefined_analysis.get("status") == "success":
            for rec in undefined_analysis.get("recommendations", []):
                if rec.get("priority") == "high":
                    high_priority.append({
                        "type": "undefined",
                        "category": rec.get("category"),
                        "message": rec.get("message"),
                        "source": "undefined_analysis"
                    })
        
        return high_priority


def main():
    """Command-line interface for the import and undefined checker."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive import and undefined variable checker"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to analyze (default: current directory)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--imports-only", action="store_true",
                       help="Check only imports")
    parser.add_argument("--undefined-only", action="store_true",
                       help="Check only undefined variables")
    parser.add_argument("--project-root", 
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = ImportAndUndefinedChecker(project_root=args.project_root)
    
    # Run checks based on arguments
    if args.imports_only:
        results = checker.check_imports(args.target)
    elif args.undefined_only:
        results = checker.check_undefined_variables(args.target)
    else:
        results = checker.run_comprehensive_check(args.target)
    
    # Save report if requested
    if args.output:
        checker.save_report(args.output)
    
    # Print high-priority issues
    high_priority = checker.get_high_priority_issues()
    if high_priority:
        print(f"\n🚨 {len(high_priority)} high-priority issues found:")
        for issue in high_priority:
            print(f"  - {issue['message']}")
    
    # Exit with appropriate code
    summary = results.get("summary", {})
    total_issues = summary.get("total_issues", 0)
    
    if total_issues == 0:
        print(f"\n✅ All checks passed!")
        return 0
    elif total_issues <= 10:
        print(f"\n⚠️  Found {total_issues} issues that need attention.")
        return 1
    else:
        print(f"\n❌ Found {total_issues} issues that require immediate attention!")
        return 2


if __name__ == "__main__":
    sys.exit(main())
