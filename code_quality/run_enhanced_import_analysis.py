#!/usr/bin/env python3
"""
Standalone Enhanced Import Analysis Runner

This script provides a simple way to run the enhanced import analysis
without complex pipeline dependencies.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.enhanced_import_analysis import (
    EnhancedImportAndUndefinedAnalyzer,
    IssueSeverity,
    IssueType
)


def main():
    """Main function for standalone enhanced import analysis."""
    parser = argparse.ArgumentParser(
        description="Enhanced Import Analysis - Standalone Runner"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to analyze (default: current directory)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--project-root", 
                       help="Project root directory (default: current directory)")
    parser.add_argument("--min-severity", choices=['low', 'medium', 'high', 'critical'], default='low',
                       help="Minimum severity level to report (default: low)")
    parser.add_argument("--max-issues-per-file", type=int, default=100,
                       help="Maximum issues to report per file (default: 100)")
    parser.add_argument("--ignore-patterns", nargs='+', 
                       default=['__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'],
                       help="Directory patterns to ignore")
    parser.add_argument("--stats", action="store_true",
                       help="Show detailed statistics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'min_severity': IssueSeverity(args.min_severity),
        'max_issues_per_file': args.max_issues_per_file,
        'ignore_patterns': args.ignore_patterns
    }
    
    # Initialize analyzer
    analyzer = EnhancedImportAndUndefinedAnalyzer(
        project_root=args.project_root,
        config=config
    )
    
    # Run analysis
    if args.verbose:
        print("Starting enhanced import analysis...")
        print(f"Target: {args.target or 'current directory'}")
        print(f"Configuration: {config}")
        print()
    
    results = analyzer.run_comprehensive_analysis(args.target)
    
    # Save report if requested
    if args.output:
        report_path = analyzer.save_report(args.output)
        if args.verbose:
            print(f"Report saved to: {report_path}")
    
    # Print high-priority issues
    high_priority = analyzer.get_high_priority_issues()
    if high_priority:
        print(f"\n🚨 {len(high_priority)} high-priority issues found:")
        for issue in high_priority:
            print(f"  - {issue['file']}:{issue['line']} - {issue['message']}")
    
    # Show detailed statistics if requested
    if args.stats:
        stats = analyzer.get_issue_statistics()
        print(f"\n📊 Detailed Statistics:")
        print(f"Import Issues:")
        print(f"  Total: {stats['import_issues']['total']}")
        print(f"  Files affected: {stats['import_issues']['files_affected']}")
        if stats['import_issues']['by_type']:
            print(f"  By type:")
            for issue_type, count in stats['import_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        if stats['import_issues']['by_severity']:
            print(f"  By severity:")
            for severity, count in stats['import_issues']['by_severity'].items():
                print(f"    {severity}: {count}")
        
        print(f"Undefined Issues:")
        print(f"  Total: {stats['undefined_issues']['total']}")
        print(f"  Files affected: {stats['undefined_issues']['files_affected']}")
        if stats['undefined_issues']['by_type']:
            print(f"  By type:")
            for issue_type, count in stats['undefined_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        if stats['undefined_issues']['by_severity']:
            print(f"  By severity:")
            for severity, count in stats['undefined_issues']['by_severity'].items():
                print(f"    {severity}: {count}")
    
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