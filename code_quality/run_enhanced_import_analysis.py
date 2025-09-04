#!/usr/bin/env python3
"""
Script to run the enhanced import analyzer on the codebase.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the code_quality directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzers.import_analyzer import ImportAnalyzer
from core.config import CodeQualityConfig


def main():
    """Run the enhanced import analysis."""
    print("🔍 Running Enhanced Import Analysis...")
    print("=" * 60)
    
    # Get the project root (parent of code_quality directory)
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Initialize configuration
    config = CodeQualityConfig()
    
    # Initialize the analyzer
    analyzer = ImportAnalyzer(config)
    
    # Analyze the entire project
    print(f"\n📁 Analyzing imports in: {project_root}")
    results = analyzer.analyze_directory(str(project_root))
    
    # Print summary
    summary = results["summary"]
    print(f"\n📊 Analysis Summary:")
    print(f"  Files analyzed: {summary['total_files_analyzed']}")
    print(f"  Total imports: {summary['total_imports']}")
    print(f"  Total issues: {summary['total_issues']}")
    print(f"  Duplicate imports: {summary['duplicate_imports']}")
    print(f"  Circular dependencies: {summary['circular_dependencies']}")
    print(f"  Conflicting imports: {summary['conflicting_imports']}")
    print(f"  Unresolvable imports: {summary['unresolvable_imports']}")
    
    # Show unresolvable imports in detail
    unresolvable = results["issues"]["unresolvable_imports"]
    if unresolvable:
        print(f"\n❌ Unresolvable Imports ({len(unresolvable)} found):")
        print("-" * 60)
        for issue in unresolvable:
            print(f"  📄 {issue['file']}:{issue['line']}")
            print(f"     {issue['message']}")
            if 'reason' in issue['details']:
                print(f"     Reason: {issue['details']['reason']}")
            print()
    else:
        print(f"\n✅ No unresolvable imports found!")
    
    # Show other critical issues
    circular = results["issues"]["circular_dependencies"]
    if circular:
        print(f"\n🔄 Circular Dependencies ({len(circular)} found):")
        print("-" * 60)
        for issue in circular[:5]:  # Show first 5
            print(f"  📄 {issue['file']}:{issue['line']}")
            print(f"     {issue['message']}")
        if len(circular) > 5:
            print(f"     ... and {len(circular) - 5} more")
        print()
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"reports/enhanced_import_analysis_{timestamp}.json"
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Return exit code based on critical issues
    critical_issues = summary['unresolvable_imports'] + summary['circular_dependencies']
    if critical_issues > 0:
        print(f"\n⚠️  Found {critical_issues} critical import issues that need attention!")
        return 1
    else:
        print(f"\n✅ No critical import issues found!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
