#!/usr/bin/env python3
"""
Simple code analysis script that runs standard tools on src/utils
"""

import subprocess
import os
import sys
from pathlib import Path
import json
import time

def run_flake8(directory):
    """Run flake8 and return results."""
    print("Running flake8...")
    try:
        result = subprocess.run(
            ["flake8", directory, "--format=json", "--max-line-length=120"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            issues = json.loads(result.stdout)
            return {"tool": "flake8", "issues_count": len(issues), "issues": issues[:10]}  # First 10 issues
        return {"tool": "flake8", "issues_count": 0, "issues": []}
    except Exception as e:
        return {"tool": "flake8", "error": str(e)}

def run_pylint(directory):
    """Run pylint and return results."""
    print("Running pylint...")
    try:
        result = subprocess.run(
            ["pylint", directory, "--output-format=json", "--max-line-length=120"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            issues = json.loads(result.stdout)
            return {"tool": "pylint", "issues_count": len(issues), "issues": issues[:10]}  # First 10 issues
        return {"tool": "pylint", "issues_count": 0, "issues": []}
    except Exception as e:
        return {"tool": "pylint", "error": str(e)}

def run_mypy(directory):
    """Run mypy and return results."""
    print("Running mypy...")
    try:
        result = subprocess.run(
            ["mypy", directory, "--ignore-missing-imports", "--no-error-summary"],
            capture_output=True,
            text=True
        )
        issues = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    issues.append(line)
        return {"tool": "mypy", "issues_count": len(issues), "issues": issues[:10]}  # First 10 issues
    except Exception as e:
        return {"tool": "mypy", "error": str(e)}

def run_bandit(directory):
    """Run bandit security analyzer."""
    print("Running bandit...")
    try:
        result = subprocess.run(
            ["bandit", "-r", directory, "-f", "json"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            data = json.loads(result.stdout)
            return {
                "tool": "bandit",
                "issues_count": len(data.get("results", [])),
                "issues": data.get("results", [])[:10]  # First 10 issues
            }
        return {"tool": "bandit", "issues_count": 0, "issues": []}
    except Exception as e:
        return {"tool": "bandit", "error": str(e)}

def run_radon_cc(directory):
    """Run radon complexity analysis."""
    print("Running radon complexity analysis...")
    try:
        result = subprocess.run(
            ["radon", "cc", directory, "-s", "-j"],
            capture_output=True,
            text=True
        )
        if result.stdout:
            data = json.loads(result.stdout)
            complex_functions = []
            for file_path, functions in data.items():
                for func in functions:
                    if func.get("complexity", 0) > 10:  # Functions with complexity > 10
                        complex_functions.append({
                            "file": file_path,
                            "function": func.get("name"),
                            "complexity": func.get("complexity"),
                            "rank": func.get("rank")
                        })
            return {
                "tool": "radon",
                "complex_functions_count": len(complex_functions),
                "complex_functions": complex_functions[:10]  # First 10
            }
        return {"tool": "radon", "complex_functions_count": 0, "complex_functions": []}
    except Exception as e:
        return {"tool": "radon", "error": str(e)}

def count_python_files(directory):
    """Count Python files in directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env']]
        count += sum(1 for f in files if f.endswith('.py'))
    return count

def main():
    """Run all analyses on src/utils."""
    directory = "src/utils"
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found!")
        return 1
    
    print(f"Analyzing {directory}...")
    print("=" * 80)
    
    # Count files
    file_count = count_python_files(directory)
    print(f"Found {file_count} Python files")
    print("=" * 80)
    
    # Run all tools
    results = []
    start_time = time.time()
    
    # Add PATH for local installations
    os.environ['PATH'] = f"/home/ubuntu/.local/bin:{os.environ['PATH']}"
    
    results.append(run_flake8(directory))
    results.append(run_pylint(directory))
    results.append(run_mypy(directory))
    results.append(run_bandit(directory))
    results.append(run_radon_cc(directory))
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total analysis time: {total_time:.2f} seconds")
    print(f"Files analyzed: {file_count}")
    print()
    
    for result in results:
        if "error" in result:
            print(f"{result['tool']}: ERROR - {result['error']}")
        else:
            if result['tool'] == 'radon':
                print(f"{result['tool']}: Found {result['complex_functions_count']} complex functions (complexity > 10)")
            else:
                print(f"{result['tool']}: Found {result['issues_count']} issues")
    
    # Save detailed results
    output_file = f"simple_analysis_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "directory": directory,
            "file_count": file_count,
            "analysis_time": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print top issues from each tool
    print("\n" + "=" * 80)
    print("TOP ISSUES BY TOOL")
    print("=" * 80)
    
    for result in results:
        if "error" not in result:
            print(f"\n{result['tool'].upper()}:")
            if result['tool'] == 'radon':
                for func in result.get('complex_functions', [])[:5]:
                    print(f"  - {func['file']}:{func['function']} - Complexity: {func['complexity']} ({func['rank']})")
            else:
                for issue in result.get('issues', [])[:5]:
                    if isinstance(issue, dict):
                        # Handle different output formats
                        if 'filename' in issue:  # flake8 format
                            print(f"  - {issue['filename']}:{issue['line_number']} - {issue['text']}")
                        elif 'path' in issue:  # pylint format
                            print(f"  - {issue['path']}:{issue.get('line', '?')} - {issue.get('message', 'No message')}")
                        elif 'filename' in issue:  # bandit format
                            print(f"  - {issue['filename']}:{issue.get('line_number', '?')} - {issue.get('issue_text', 'No text')}")
                    else:
                        # String format (mypy)
                        print(f"  - {issue}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())