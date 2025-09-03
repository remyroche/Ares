#!/usr/bin/env python3
"""Analyze syntax errors from the sequential fixer output."""

import re
from pathlib import Path
from collections import defaultdict
import json

def parse_error_log(log_text):
    """Parse the error log and extract syntax errors."""
    errors = defaultdict(list)
    error_pattern = r"Error parsing (.*?): (.*?) \(.*?, line (\d+)\)"
    
    for match in re.finditer(error_pattern, log_text):
        file_path = match.group(1)
        error_msg = match.group(2)
        line_num = int(match.group(3))
        errors[file_path].append({'line': line_num, 'error': error_msg})
    
    # Extract warnings about syntax errors after fixing
    warning_pattern = r"Warning: \d+ files have syntax errors after fixing:.*?(?=Processing|$)"
    for warning in re.findall(warning_pattern, log_text, re.DOTALL):
        file_pattern = r"- (.*?)(?:\n|$)"
        for file_match in re.finditer(file_pattern, warning):
            file_path = file_match.group(1).strip()
            if file_path not in errors:
                errors[file_path].append({'line': 0, 'error': 'Syntax error after auto-fixing'})
    
    return errors

def categorize_errors(errors):
    """Categorize errors by type."""
    categories = defaultdict(list)
    
    for file_path, file_errors in errors.items():
        for error in file_errors:
            error_msg = error['error']
            
            if 'unexpected indent' in error_msg:
                categories['indentation'].append(file_path)
            elif 'unmatched' in error_msg:
                categories['unmatched_brackets'].append(file_path)
            elif 'invalid syntax' in error_msg:
                categories['invalid_syntax'].append(file_path)
            elif 'expected' in error_msg and 'block' in error_msg:
                categories['missing_block'].append(file_path)
            elif 'unterminated string' in error_msg:
                categories['unterminated_string'].append(file_path)
            elif 'unindent does not match' in error_msg:
                categories['indentation_mismatch'].append(file_path)
            else:
                categories['other'].append(file_path)
    
    # Remove duplicates
    for category in categories:
        categories[category] = list(set(categories[category]))
    
    return categories

def generate_report(errors, categories):
    """Generate a detailed report."""
    report = []
    report.append("# Syntax Error Analysis Report")
    report.append(f"\nTotal files with syntax errors: {len(errors)}")
    report.append("\n## Error Categories:")
    
    for category, files in sorted(categories.items()):
        report.append(f"\n### {category.replace('_', ' ').title()} ({len(files)} files)")
        for file in sorted(files)[:10]:
            report.append(f"- {file}")
        if len(files) > 10:
            report.append(f"- ... and {len(files) - 10} more")
    
    report.append("\n## Priority Files to Fix:")
    report.append("\nThese are critical files that should be fixed first:")
    
    priority_files = [
        "src/exchange/binance.py",
        "src/training/training_manager.py",
        "src/training/model_trainer.py",
        "src/analyst/analyst.py",
        "src/utils/model_manager.py",
        "src/config/config.py",
        "src/training/steps/step1/missing_data_downloader_and_gap_filler.py"
    ]
    
    for file in priority_files:
        if file in errors:
            report.append(f"\n### {file}")
            for error in errors[file]:
                report.append(f"- Line {error['line']}: {error['error']}")
    
    return "\n".join(report)

def main():
    # Read from saved file
    with open("/workspace/syntax_errors_from_log.txt", "r") as f:
        log_text = f.read()
    
    errors = parse_error_log(log_text)
    categories = categorize_errors(errors)
    report = generate_report(errors, categories)
    
    # Save report
    with open("/workspace/syntax_error_analysis_report.md", "w") as f:
        f.write(report)
    
    # Save JSON
    with open("/workspace/syntax_errors.json", "w") as f:
        json.dump({'total_files': len(errors), 'errors': errors, 'categories': categories}, f, indent=2)
    
    print(f"Analysis complete. Found {len(errors)} files with syntax errors.")
    print(f"Report saved to: syntax_error_analysis_report.md")

if __name__ == "__main__":
    main()