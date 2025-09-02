#!/usr/bin/env python3
"""
Comprehensive Dead Code Detection Script

This script uses all available analyzers to detect:
- Dead/unused code (functions, classes, methods)
- Unused imports
- Unused dependencies
- Functions that are never called
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set

# Add code_quality to the path
sys.path.insert(0, str(Path(__file__).parent))

from code_quality.analyzers.dead_code_analyzer import DeadCodeAnalyzer
from code_quality.analyzers.import_analyzer import ImportAnalyzer
from code_quality.analyzers.call_graph_analyzer import CallGraphAnalyzer
from code_quality.analyzers.dependency_analyzer import DependencyAnalyzer
from code_quality.analyzers.signature_analyzer import SignatureAnalyzer
from code_quality.core.config import get_default_config


class ComprehensiveDeadCodeDetector:
    """Combines multiple analyzers to detect all forms of dead/unused code."""
    
    def __init__(self, target_path: str):
        self.target_path = Path(target_path).resolve()
        self.config = get_default_config()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "target_path": str(self.target_path),
            "dead_code": {},
            "unused_imports": {},
            "unused_functions": {},
            "unused_dependencies": {},
            "summary": {}
        }
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run all analyzers and compile results."""
        print(f"Running comprehensive dead code analysis on: {self.target_path}\n")
        
        # 1. Run dead code analyzer (using Vulture)
        print("1. Running dead code analysis (Vulture)...")
        self._run_dead_code_analysis()
        
        # 2. Run import analysis
        print("\n2. Running import analysis...")
        self._run_import_analysis()
        
        # 3. Run call graph analysis
        print("\n3. Running call graph analysis...")
        self._run_call_graph_analysis()
        
        # 4. Run dependency analysis
        print("\n4. Running dependency analysis...")
        self._run_dependency_analysis()
        
        # 5. Run signature analysis for unused functions
        print("\n5. Running signature analysis...")
        self._run_signature_analysis()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _run_dead_code_analysis(self):
        """Use DeadCodeAnalyzer to find dead code."""
        try:
            analyzer = DeadCodeAnalyzer(self.config)
            
            if self.target_path.is_file():
                report = analyzer.analyze_file(str(self.target_path))
            else:
                report = analyzer.analyze_directory(str(self.target_path))
            
            # Extract issues by type
            dead_code_by_type = {}
            for issue_type, count in report.issues_by_type.items():
                if count > 0:
                    dead_code_by_type[issue_type] = {
                        "count": count,
                        "files": []
                    }
            
            # Group issues by file
            for file_path, issues in report.issues_by_file.items():
                for issue in issues:
                    if issue.issue_type not in dead_code_by_type:
                        dead_code_by_type[issue.issue_type] = {
                            "count": 0,
                            "files": []
                        }
                    
                    dead_code_by_type[issue.issue_type]["files"].append({
                        "file": file_path,
                        "line": issue.line_number,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity
                    })
            
            self.results["dead_code"] = {
                "total_issues": report.total_issues,
                "issues_by_type": dead_code_by_type,
                "confidence_distribution": report.confidence_distribution,
                "potential_lines_to_remove": sum(report.potential_savings.values())
            }
            
            print(f"  Found {report.total_issues} potential dead code issues")
            
        except Exception as e:
            print(f"  Error in dead code analysis: {e}")
            self.results["dead_code"] = {"error": str(e)}
    
    def _run_import_analysis(self):
        """Analyze imports to find unused ones."""
        try:
            analyzer = ImportAnalyzer(self.config)
            results = analyzer.analyze_directory(str(self.target_path))
            
            # Extract unused imports
            unused_imports = []
            if "unused_imports" in results:
                for import_info in results["unused_imports"]:
                    unused_imports.append({
                        "file": import_info.file_path,
                        "line": import_info.line_number,
                        "import": import_info.message,
                        "severity": import_info.severity
                    })
            
            self.results["unused_imports"] = {
                "total": len(unused_imports),
                "imports": unused_imports
            }
            
            print(f"  Found {len(unused_imports)} unused imports")
            
        except Exception as e:
            print(f"  Error in import analysis: {e}")
            self.results["unused_imports"] = {"error": str(e)}
    
    def _run_call_graph_analysis(self):
        """Analyze call graph to find uncalled functions."""
        try:
            analyzer = CallGraphAnalyzer(self.config)
            results = analyzer.analyze_directory(str(self.target_path))
            
            # Find functions that are never called
            dead_code_candidates = []
            if "dead_code_candidates" in results:
                for func_name, func_info in results["dead_code_candidates"].items():
                    dead_code_candidates.append({
                        "function": func_name,
                        "file": func_info.get("file", "unknown"),
                        "line": func_info.get("line", 0),
                        "type": func_info.get("type", "function"),
                        "reason": "Never called in the codebase"
                    })
            
            self.results["unused_functions"]["from_call_graph"] = {
                "total": len(dead_code_candidates),
                "functions": dead_code_candidates
            }
            
            print(f"  Found {len(dead_code_candidates)} potentially unused functions")
            
        except Exception as e:
            print(f"  Error in call graph analysis: {e}")
            self.results["unused_functions"]["from_call_graph"] = {"error": str(e)}
    
    def _run_dependency_analysis(self):
        """Analyze dependencies to find unused ones."""
        try:
            analyzer = DependencyAnalyzer(self.config)
            results = analyzer.analyze_directory(str(self.target_path))
            
            # Extract unused dependencies
            unused_deps = []
            if "unused_dependencies" in results:
                for dep in results["unused_dependencies"]:
                    unused_deps.append({
                        "name": dep["name"],
                        "version": dep.get("version", "unknown"),
                        "source": dep.get("source", "unknown"),
                        "reason": "Not imported anywhere in the codebase"
                    })
            
            self.results["unused_dependencies"] = {
                "total": len(unused_deps),
                "dependencies": unused_deps
            }
            
            print(f"  Found {len(unused_deps)} unused dependencies")
            
        except Exception as e:
            print(f"  Error in dependency analysis: {e}")
            self.results["unused_dependencies"] = {"error": str(e)}
    
    def _run_signature_analysis(self):
        """Analyze function signatures to find unused ones."""
        try:
            analyzer = SignatureAnalyzer(self.config)
            results = analyzer.analyze_directory(str(self.target_path))
            
            # Extract unused functions
            unused_funcs = []
            if "unused_functions" in results:
                for func_name, func_info in results["unused_functions"].items():
                    unused_funcs.append({
                        "function": func_name,
                        "file": func_info.get("file", "unknown"),
                        "line": func_info.get("line", 0),
                        "signature": func_info.get("signature", "unknown"),
                        "reason": "No calls found to this function"
                    })
            
            self.results["unused_functions"]["from_signature"] = {
                "total": len(unused_funcs),
                "functions": unused_funcs
            }
            
            print(f"  Found {len(unused_funcs)} unused functions via signature analysis")
            
        except Exception as e:
            print(f"  Error in signature analysis: {e}")
            self.results["unused_functions"]["from_signature"] = {"error": str(e)}
    
    def _generate_summary(self):
        """Generate a summary of all findings."""
        summary = {
            "total_dead_code_issues": 0,
            "total_unused_imports": 0,
            "total_unused_functions": 0,
            "total_unused_dependencies": 0,
            "total_issues": 0
        }
        
        # Count dead code issues
        if "total_issues" in self.results.get("dead_code", {}):
            summary["total_dead_code_issues"] = self.results["dead_code"]["total_issues"]
        
        # Count unused imports
        if "total" in self.results.get("unused_imports", {}):
            summary["total_unused_imports"] = self.results["unused_imports"]["total"]
        
        # Count unused functions (combine both analyzers)
        for analyzer_key in ["from_call_graph", "from_signature"]:
            if analyzer_key in self.results.get("unused_functions", {}):
                if "total" in self.results["unused_functions"][analyzer_key]:
                    summary["total_unused_functions"] += self.results["unused_functions"][analyzer_key]["total"]
        
        # Count unused dependencies
        if "total" in self.results.get("unused_dependencies", {}):
            summary["total_unused_dependencies"] = self.results["unused_dependencies"]["total"]
        
        # Total issues
        summary["total_issues"] = (
            summary["total_dead_code_issues"] +
            summary["total_unused_imports"] +
            summary["total_unused_functions"] +
            summary["total_unused_dependencies"]
        )
        
        self.results["summary"] = summary
    
    def save_report(self, output_path: str = None):
        """Save the analysis report to a file."""
        if output_path is None:
            output_path = f"dead_code_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a summary of findings to console."""
        print("\n" + "="*60)
        print("DEAD CODE DETECTION SUMMARY")
        print("="*60)
        
        summary = self.results["summary"]
        print(f"Total issues found: {summary['total_issues']}")
        print(f"  - Dead code issues: {summary['total_dead_code_issues']}")
        print(f"  - Unused imports: {summary['total_unused_imports']}")
        print(f"  - Unused functions: {summary['total_unused_functions']}")
        print(f"  - Unused dependencies: {summary['total_unused_dependencies']}")
        
        # Print some examples if available
        if self.results["dead_code"].get("issues_by_type"):
            print("\nDead code by type:")
            for issue_type, data in self.results["dead_code"]["issues_by_type"].items():
                print(f"  - {issue_type}: {data['count']} occurrences")
        
        if self.results["unused_imports"].get("imports"):
            print(f"\nExample unused imports (showing first 5):")
            for imp in self.results["unused_imports"]["imports"][:5]:
                print(f"  - {imp['file']}:{imp['line']} - {imp['import']}")
        
        if self.results["unused_functions"].get("from_call_graph", {}).get("functions"):
            print(f"\nExample unused functions (showing first 5):")
            for func in self.results["unused_functions"]["from_call_graph"]["functions"][:5]:
                print(f"  - {func['file']}:{func['line']} - {func['function']}()")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive dead code detection for Python projects"
    )
    parser.add_argument(
        "path",
        help="Path to Python file or directory to analyze"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path for the JSON report",
        default=None
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Validate path
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: Path '{args.path}' does not exist")
        return 1
    
    # Run analysis
    detector = ComprehensiveDeadCodeDetector(args.path)
    results = detector.run_analysis()
    
    # Print summary
    detector.print_summary()
    
    # Save report
    output_path = detector.save_report(args.output)
    
    # Return exit code based on findings
    if results["summary"]["total_issues"] > 0:
        return 1  # Found issues
    return 0  # No issues found


if __name__ == "__main__":
    sys.exit(main())