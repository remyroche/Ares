#!/usr/bin/env python3
"""
Run Sequential Fixer without external dependencies
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add the workspace to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_python_files(directory, exclude_patterns=None):
    """Find all Python files in a directory."""
    exclude_patterns = exclude_patterns or []
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def run_basic_analysis(target_dir):
    """Run basic code analysis without external dependencies."""
    print(f"\n{'='*70}")
    print("SEQUENTIAL CODE ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"Target directory: {target_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all Python files
    python_files = find_python_files(target_dir, exclude_patterns=['__pycache__', '.git', 'venv'])
    print(f"\nTotal Python files found: {len(python_files)}")
    
    results = {
        "target": target_dir,
        "timestamp": datetime.now().isoformat(),
        "total_files": len(python_files),
        "files_analyzed": 0,
        "syntax_errors": [],
        "import_errors": [],
        "indentation_errors": [],
        "other_errors": [],
        "clean_files": [],
        "file_details": {}
    }
    
    print(f"\n{'='*50}")
    print("ANALYZING FILES")
    print(f"{'='*50}")
    
    for i, file_path in enumerate(python_files, 1):
        relative_path = os.path.relpath(file_path, target_dir)
        print(f"\n[{i}/{len(python_files)}] Analyzing: {relative_path}")
        
        file_info = {
            "path": file_path,
            "relative_path": relative_path,
            "size": os.path.getsize(file_path),
            "lines": 0,
            "errors": []
        }
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                file_info["lines"] = len(lines)
            
            # Try to compile the file
            try:
                compile(content, file_path, 'exec')
                print(f"  ✓ Syntax: Valid")
                
                # Basic import analysis
                import_count = 0
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        import_count += 1
                
                print(f"  ✓ Imports: {import_count} found")
                
                # Check for basic issues
                issues = []
                
                # Check for tabs vs spaces
                has_tabs = '\t' in content
                has_spaces = any(line.startswith(' ') for line in lines if line.strip())
                if has_tabs and has_spaces:
                    issues.append("Mixed tabs and spaces for indentation")
                
                # Check for trailing whitespace
                trailing_whitespace_lines = [i+1 for i, line in enumerate(lines) if line.rstrip() != line]
                if trailing_whitespace_lines:
                    issues.append(f"Trailing whitespace on {len(trailing_whitespace_lines)} lines")
                
                # Check for very long lines
                long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
                if long_lines:
                    issues.append(f"{len(long_lines)} lines exceed 120 characters")
                
                if issues:
                    print(f"  ! Style issues: {len(issues)}")
                    for issue in issues:
                        print(f"    - {issue}")
                    file_info["errors"].extend(issues)
                else:
                    print(f"  ✓ Style: Clean")
                    results["clean_files"].append(relative_path)
                
            except SyntaxError as e:
                error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
                print(f"  ✗ Syntax Error: {error_msg}")
                file_info["errors"].append(error_msg)
                results["syntax_errors"].append({
                    "file": relative_path,
                    "line": e.lineno,
                    "error": e.msg
                })
            
            except IndentationError as e:
                error_msg = f"IndentationError at line {e.lineno}: {e.msg}"
                print(f"  ✗ Indentation Error: {error_msg}")
                file_info["errors"].append(error_msg)
                results["indentation_errors"].append({
                    "file": relative_path,
                    "line": e.lineno,
                    "error": e.msg
                })
            
        except Exception as e:
            error_msg = f"Failed to read file: {str(e)}"
            print(f"  ✗ Error: {error_msg}")
            file_info["errors"].append(error_msg)
            results["other_errors"].append({
                "file": relative_path,
                "error": str(e)
            })
        
        results["file_details"][relative_path] = file_info
        results["files_analyzed"] += 1
    
    # Generate summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nFiles analyzed: {results['files_analyzed']}/{results['total_files']}")
    print(f"Clean files: {len(results['clean_files'])}")
    print(f"Files with syntax errors: {len(results['syntax_errors'])}")
    print(f"Files with indentation errors: {len(results['indentation_errors'])}")
    print(f"Files with other errors: {len(results['other_errors'])}")
    
    # Calculate total issues
    total_issues = 0
    for file_info in results["file_details"].values():
        total_issues += len(file_info["errors"])
    
    print(f"\nTotal issues found: {total_issues}")
    
    # Top problematic files
    problematic_files = [
        (path, info) for path, info in results["file_details"].items() 
        if info["errors"]
    ]
    problematic_files.sort(key=lambda x: len(x[1]["errors"]), reverse=True)
    
    if problematic_files:
        print(f"\nTop 10 most problematic files:")
        for i, (path, info) in enumerate(problematic_files[:10], 1):
            print(f"  {i}. {path} - {len(info['errors'])} issues")
    
    # Directory summary
    dir_issues = {}
    for path, info in results["file_details"].items():
        dir_name = os.path.dirname(path) or "."
        if dir_name not in dir_issues:
            dir_issues[dir_name] = {"files": 0, "issues": 0}
        dir_issues[dir_name]["files"] += 1
        dir_issues[dir_name]["issues"] += len(info["errors"])
    
    print(f"\nIssues by directory:")
    sorted_dirs = sorted(dir_issues.items(), key=lambda x: x[1]["issues"], reverse=True)
    for dir_name, stats in sorted_dirs[:10]:
        if stats["issues"] > 0:
            print(f"  {dir_name}: {stats['issues']} issues in {stats['files']} files")
    
    return results

def save_report(results, output_dir):
    """Save the analysis report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"sequential_analysis_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report saved: {json_path}")
    
    # Save text report
    text_path = os.path.join(output_dir, f"sequential_analysis_{timestamp}.txt")
    with open(text_path, 'w') as f:
        f.write("SEQUENTIAL CODE ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Target: {results['target']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Total files: {results['total_files']}\n")
        f.write(f"Files analyzed: {results['files_analyzed']}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*50 + "\n")
        f.write(f"Clean files: {len(results['clean_files'])}\n")
        f.write(f"Syntax errors: {len(results['syntax_errors'])}\n")
        f.write(f"Indentation errors: {len(results['indentation_errors'])}\n")
        f.write(f"Other errors: {len(results['other_errors'])}\n\n")
        
        if results['syntax_errors']:
            f.write("SYNTAX ERRORS\n")
            f.write("-"*50 + "\n")
            for error in results['syntax_errors']:
                f.write(f"- {error['file']} (line {error['line']}): {error['error']}\n")
            f.write("\n")
        
        if results['indentation_errors']:
            f.write("INDENTATION ERRORS\n")
            f.write("-"*50 + "\n")
            for error in results['indentation_errors']:
                f.write(f"- {error['file']} (line {error['line']}): {error['error']}\n")
            f.write("\n")
    
    print(f"Text report saved: {text_path}")
    
    # Save markdown report
    md_path = os.path.join(output_dir, f"sequential_analysis_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write("# Sequential Code Analysis Report\n\n")
        f.write(f"**Target:** `{results['target']}`  \n")
        f.write(f"**Timestamp:** {results['timestamp']}  \n")
        f.write(f"**Total files:** {results['total_files']}  \n")
        f.write(f"**Files analyzed:** {results['files_analyzed']}  \n\n")
        
        f.write("## Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Clean files | {len(results['clean_files'])} |\n")
        f.write(f"| Syntax errors | {len(results['syntax_errors'])} |\n")
        f.write(f"| Indentation errors | {len(results['indentation_errors'])} |\n")
        f.write(f"| Other errors | {len(results['other_errors'])} |\n\n")
        
        # Add top issues
        problematic_files = [
            (path, info) for path, info in results["file_details"].items() 
            if info["errors"]
        ]
        problematic_files.sort(key=lambda x: len(x[1]["errors"]), reverse=True)
        
        if problematic_files:
            f.write("## Top 10 Most Problematic Files\n\n")
            f.write("| File | Issues |\n")
            f.write("|------|--------|\n")
            for path, info in problematic_files[:10]:
                f.write(f"| `{path}` | {len(info['errors'])} |\n")
            f.write("\n")
        
        if results['syntax_errors']:
            f.write("## Syntax Errors\n\n")
            for error in results['syntax_errors'][:20]:  # Limit to first 20
                f.write(f"- **{error['file']}** (line {error['line']}): {error['error']}\n")
            if len(results['syntax_errors']) > 20:
                f.write(f"\n... and {len(results['syntax_errors']) - 20} more syntax errors\n")
            f.write("\n")
    
    print(f"Markdown report saved: {md_path}")

if __name__ == "__main__":
    # Run analysis on src directory
    target_dir = os.path.join(os.path.dirname(__file__), "src")
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory '{target_dir}' not found!")
        sys.exit(1)
    
    print(f"Starting analysis of: {target_dir}")
    start_time = time.time()
    
    # Run the analysis
    results = run_basic_analysis(target_dir)
    
    # Save reports
    output_dir = os.path.join(os.path.dirname(__file__), "sequential_fixer_reports")
    save_report(results, output_dir)
    
    duration = time.time() - start_time
    print(f"\nAnalysis completed in {duration:.2f} seconds")