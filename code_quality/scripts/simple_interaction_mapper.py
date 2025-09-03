#!/usr/bin/env python3
"""
Simple Code Interaction Mapper

This script uses the existing comprehensive code review tool to map code interactions.
It extracts and visualizes:
- Function call relationships
- Module dependencies (through imports)
- Async/await patterns
- Function definitions and their usage
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_code_review import CodeQualityReviewer


def extract_interactions(report_data):
    """Extract interaction data from the comprehensive review report."""
    interactions = {
        'module_imports': defaultdict(list),
        'function_calls': defaultdict(list),
        'function_definitions': {},
        'async_patterns': [],
        'undefined_functions': [],
        'import_graph': defaultdict(set),
        'call_graph': defaultdict(set)
    }
    
    # Process issues to find patterns
    for issue in report_data.get('issues', []):
        if issue['issue_type'] == 'undefined_function':
            interactions['undefined_functions'].append({
                'function': issue['message'].split("'")[1],
                'file': issue['file_path'],
                'line': issue['line_number']
            })
        elif issue['issue_type'] == 'missing_await':
            interactions['async_patterns'].append({
                'function': issue['message'].split("'")[1],
                'file': issue['file_path'],
                'line': issue['line_number'],
                'issue': 'missing_await'
            })
    
    return interactions


def generate_interaction_summary(project_root, output_dir=None):
    """Generate a summary of code interactions using the comprehensive review tool."""
    print(f"\n{'='*60}")
    print("CODE INTERACTION MAPPER")
    print(f"{'='*60}\n")
    
    # Initialize the comprehensive reviewer
    print("[1/3] Initializing code reviewer...")
    reviewer = CodeQualityReviewer(project_root)
    
    # Run the analysis
    print("[2/3] Analyzing code interactions...")
    report_data = reviewer.scan_project()
    
    # Extract interactions
    interactions = extract_interactions(report_data)
    
    # Prepare output directory
    if not output_dir:
        output_dir = Path(project_root) / "code_quality" / "interaction_maps"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full report
    full_report_file = output_dir / f"full_analysis_{timestamp}.json"
    with open(full_report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate interaction summary
    summary_file = output_dir / f"interaction_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("CODE INTERACTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Project overview
        f.write("PROJECT OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"Project Root: {project_root}\n")
        f.write(f"Files Analyzed: {report_data['summary']['files_processed']}\n")
        f.write(f"Total Issues: {report_data['summary']['total_issues']}\n")
        f.write(f"Function Analysis:\n")
        f.write(f"  - Total Function Calls: {report_data['function_analysis']['total_calls']}\n")
        f.write(f"  - Total Function Definitions: {report_data['function_analysis']['total_definitions']}\n")
        f.write(f"  - Async Functions: {report_data['function_analysis']['async_functions']}\n\n")
        
        # Issue breakdown
        f.write("INTERACTION ISSUES\n")
        f.write("-" * 30 + "\n")
        
        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in report_data['issues']:
            issues_by_type[issue['issue_type']].append(issue)
        
        # Focus on interaction-related issues
        interaction_types = ['undefined_function', 'missing_await', 'import_error', 
                           'circular_import', 'parameter_mismatch']
        
        for issue_type in interaction_types:
            if issue_type in issues_by_type:
                f.write(f"\n{issue_type.replace('_', ' ').upper()} ({len(issues_by_type[issue_type])}):\n")
                for issue in issues_by_type[issue_type][:10]:  # Show first 10
                    f.write(f"  - {issue['file_path']}:{issue['line_number']} - {issue['message']}\n")
                if len(issues_by_type[issue_type]) > 10:
                    f.write(f"  ... and {len(issues_by_type[issue_type]) - 10} more\n")
        
        # Undefined functions summary
        if interactions['undefined_functions']:
            f.write("\n\nUNDEFINED FUNCTIONS SUMMARY\n")
            f.write("-" * 30 + "\n")
            unique_functions = set(item['function'] for item in interactions['undefined_functions'])
            f.write(f"Total unique undefined functions: {len(unique_functions)}\n")
            f.write("Most common undefined functions:\n")
            func_counts = defaultdict(int)
            for item in interactions['undefined_functions']:
                func_counts[item['function']] += 1
            for func, count in sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"  - {func}: {count} occurrences\n")
        
        # Async patterns
        if interactions['async_patterns']:
            f.write("\n\nASYNC/AWAIT PATTERNS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total async issues: {len(interactions['async_patterns'])}\n")
            for pattern in interactions['async_patterns'][:10]:
                f.write(f"  - {pattern['file']}:{pattern['line']} - {pattern['function']} needs await\n")
    
    # Generate a simple visualization script
    viz_script = output_dir / f"visualize_interactions_{timestamp}.py"
    with open(viz_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Simple visualization script for code interactions.
Run this to generate graphical representations of the interactions.
\"\"\"

import json
from pathlib import Path

# Load the analysis data
with open('""" + str(full_report_file) + """', 'r') as f:
    data = json.load(f)

print("VISUALIZATION GUIDE")
print("=" * 50)
print()
print("The analysis found:")
print(f"- {data['summary']['files_processed']} files processed")
print(f"- {data['summary']['total_issues']} total issues")
print(f"- {data['function_analysis']['total_calls']} function calls")
print(f"- {data['function_analysis']['total_definitions']} function definitions")
print()
print("To visualize the interactions:")
print("1. Use a tool like Graphviz to create dependency graphs")
print("2. Use network visualization libraries (networkx, pyvis) for interactive graphs")
print("3. Create call flow diagrams from the function call data")
print()
print("Key insights:")
issues_by_type = {}
for issue in data['issues']:
    issue_type = issue['issue_type']
    if issue_type not in issues_by_type:
        issues_by_type[issue_type] = 0
    issues_by_type[issue_type] += 1

for issue_type, count in sorted(issues_by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"- {issue_type}: {count} occurrences")
""")
    
    print(f"\n[3/3] Generating reports...")
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")
    
    print("Generated files:")
    print(f"  1. Full analysis report: {full_report_file}")
    print(f"  2. Interaction summary: {summary_file}")
    print(f"  3. Visualization guide: {viz_script}")
    
    print("\nKey findings:")
    print(f"  - Files processed: {report_data['summary']['files_processed']}")
    print(f"  - Total issues found: {report_data['summary']['total_issues']}")
    print(f"  - Undefined functions: {len(interactions['undefined_functions'])}")
    print(f"  - Async/await issues: {len(interactions['async_patterns'])}")
    
    return {
        'full_report': str(full_report_file),
        'summary': str(summary_file),
        'visualization': str(viz_script),
        'data': report_data
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Code Interaction Mapper')
    parser.add_argument('--project-root', default='/workspace', 
                       help='Root directory of the project to analyze')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    results = generate_interaction_summary(args.project_root, args.output_dir)
    
    print("\nTo explore the results further:")
    print(f"  1. View the summary: cat {results['summary']}")
    print(f"  2. Analyze the JSON data: python3 {results['visualization']}")
    print(f"  3. Process the full report: jq . {results['full_report']}")


if __name__ == '__main__':
    main()