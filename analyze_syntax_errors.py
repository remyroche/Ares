#!/usr/bin/env python3
"""Analyze syntax errors from the sequential fixer reports."""

import json
import os
from collections import defaultdict
from pathlib import Path


def analyze_syntax_errors():
    """Analyze syntax errors from all sequential fixer reports."""
    reports_dir = Path("/workspace/sequential_fixer_reports")
    
    # Find the most recent import analysis reports
    import_reports = list(reports_dir.glob("import_analysis_report_*.json"))
    
    all_errors = defaultdict(list)
    file_error_counts = defaultdict(int)
    error_type_counts = defaultdict(int)
    
    for report_file in import_reports:
        print(f"\nAnalyzing {report_file.name}...")
        try:
            with open(report_file) as f:
                data = json.load(f)
            
            # The errors are in the parsing errors section
            results = data.get("results", {})
            full_results = results.get("full_results", {})
            
            # Count parsing errors
            parsing_errors = full_results.get("parsing_errors", [])
            print(f"Found {len(parsing_errors)} parsing errors")
            
            for error in parsing_errors:
                file_path = error.get("file", "unknown")
                error_msg = error.get("error", "unknown error")
                
                all_errors[file_path].append(error_msg)
                file_error_counts[file_path] += 1
                
                # Extract error type from message
                if "unexpected indent" in error_msg:
                    error_type_counts["unexpected_indent"] += 1
                elif "invalid syntax" in error_msg:
                    error_type_counts["invalid_syntax"] += 1
                elif "expected 'except' or 'finally'" in error_msg:
                    error_type_counts["missing_except_finally"] += 1
                elif "unmatched" in error_msg:
                    error_type_counts["unmatched_bracket"] += 1
                elif "unindent does not match" in error_msg:
                    error_type_counts["indentation_mismatch"] += 1
                elif "expected an indented block" in error_msg:
                    error_type_counts["missing_indented_block"] += 1
                elif "unterminated string literal" in error_msg:
                    error_type_counts["unterminated_string"] += 1
                else:
                    error_type_counts["other"] += 1
                    
        except Exception as e:
            print(f"Error reading {report_file}: {e}")
    
    # Sort files by error count
    sorted_files = sorted(file_error_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("SYNTAX ERROR ANALYSIS")
    print("="*80)
    
    print(f"\nTotal files with syntax errors: {len(file_error_counts)}")
    print(f"Total syntax errors: {sum(file_error_counts.values())}")
    
    print("\nError types distribution:")
    for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")
    
    print("\nTop 20 files with most syntax errors:")
    for i, (file_path, count) in enumerate(sorted_files[:20], 1):
        print(f"{i:2}. {file_path}: {count} errors")
        # Show first error for each file
        if file_path in all_errors and all_errors[file_path]:
            print(f"    First error: {all_errors[file_path][0]}")
    
    # Write detailed report
    with open("/workspace/syntax_errors_detailed.json", "w") as f:
        json.dump({
            "total_files": len(file_error_counts),
            "total_errors": sum(file_error_counts.values()),
            "error_types": dict(error_type_counts),
            "files": dict(all_errors)
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: /workspace/syntax_errors_detailed.json")
    
    return sorted_files


if __name__ == "__main__":
    analyze_syntax_errors()