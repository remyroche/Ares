#!/usr/bin/env python3
"""
Static Analysis Runner
Uses the StaticAnalysisAnalyzer to systematically analyze and fix code quality issues.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# Add the code_quality directory to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))

try:
    from analyzers.static_analysis_analyzer import StaticAnalysisAnalyzer
except ImportError:
    print("Could not import StaticAnalysisAnalyzer. Please ensure it's in the correct location.")
    sys.exit(1)

@dataclass
class CodeQualityConfig:
    """Simple configuration class for the analyzer."""
    analysis_timeout: int = 300
    max_line_length: int = 120
    ignore_patterns: List[str] = None

    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = ["__pycache__", "*.pyc", ".git", "node_modules"]

def main():
    """Run comprehensive static analysis on the codebase."""
    print("🚀 Starting comprehensive static analysis...")

    # Create configuration
    config = CodeQualityConfig()

    # Initialize analyzer
    analyzer = StaticAnalysisAnalyzer(config)

    # Define directories to analyze
    base_path = Path(__file__).parent
    directories_to_analyze = [
        base_path / "src",
        # Add other directories as needed
    ]

    all_results = {
        "analysis_summary": {},
        "directory_results": {},
        "total_files": 0,
        "total_issues": 0,
        "critical_issues": 0,
        "warnings": 0,
        "security_issues": 0
    }

    # Analyze each directory
    for directory in directories_to_analyze:
        if directory.exists():
            print(f"\n📁 Analyzing directory: {directory}")
            try:
                results = analyzer.analyze_directory(str(directory))
                all_results["directory_results"][str(directory)] = results

                # Update totals
                dir_summary = results.get("summary", {})
                all_results["total_files"] += dir_summary.get("total_files", 0)
                all_results["total_issues"] += dir_summary.get("total_issues", 0)
                all_results["critical_issues"] += dir_summary.get("critical_issues", 0)
                all_results["warnings"] += dir_summary.get("warnings", 0)
                all_results["security_issues"] += dir_summary.get("security_issues", 0)

                print(f"   Files: {dir_summary.get('total_files', 0)}")
                print(f"   Issues: {dir_summary.get('total_issues', 0)}")
                print(f"   Critical: {dir_summary.get('critical_issues', 0)}")
                print(f"   Warnings: {dir_summary.get('warnings', 0)}")
                print(f"   Security: {dir_summary.get('security_issues', 0)}")

            except Exception as e:
                print(f"❌ Error analyzing {directory}: {str(e)}")
                all_results["directory_results"][str(directory)] = {
                    "error": str(e),
                    "status": "failed"
                }

    # Save results
    output_file = base_path / "static_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n📊 Analysis Summary:")
    print(f"   Total Files: {all_results['total_files']}")
    print(f"   Total Issues: {all_results['total_issues']}")
    print(f"   Critical Issues: {all_results['critical_issues']}")
    print(f"   Warnings: {all_results['warnings']}")
    print(f"   Security Issues: {all_results['security_issues']}")

    print(f"\n💾 Results saved to: {output_file}")

    # Print top issues by file
    print("\n🔍 Top files with most issues:")
    file_issues = []
    for dir_path, dir_results in all_results["directory_results"].items():
        if "files" in dir_results:
            for file_path, file_result in dir_results["files"].items():
                summary = file_result.get("summary", {})
                total_issues = summary.get("total_issues", 0)
                critical_issues = summary.get("critical_issues", 0)
                if total_issues > 0:
                    file_issues.append((file_path, total_issues, critical_issues))

    # Sort by total issues (descending)
    file_issues.sort(key=lambda x: x[1], reverse=True)

    for file_path, total_issues, critical_issues in file_issues[:10]:  # Top 10
        print(f"   {Path(file_path).name}: {total_issues} issues ({critical_issues} critical)")

    return all_results

if __name__ == "__main__":
    main()
