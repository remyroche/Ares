#!/usr/bin/env python3
"""
Quick Error Scanner - Fast overview of syntax errors

This script provides a quick scan and summary of files with syntax errors,
focusing on the most important information for prioritization.
"""

import os
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def quick_scan() -> Dict:
    """Perform a quick scan of all Python files."""
    print("🔍 Quick scanning for syntax errors...")
    
    # Find all Python files
    python_files=[]
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Quick scan using subprocess
    cmd=f"find . -name '*.py' -type f -exec python -m py_compile {{}} \\; 2>&1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse results
    error_lines=result.stderr.split('\n')
    error_files=defaultdict(list)
    error_types=Counter()
    
    for line in error_lines:
        if line.strip() and ('SyntaxError' in line or 'IndentationError' in line):
            # Extract file path
            if 'File "' in line:
                file_match=line.split('File "')[1].split('"')[0]
                file_path=os.path.relpath(file_match, '.')
            else:
                # Try to extract from the beginning of the line
                parts=line.split(':')
                if len(parts) >= 2:
                    file_path=parts[0].strip()
                else:
                    continue
            
            # Count error types
            if 'SyntaxError' in line:
                error_types['SyntaxError'] += 1
            elif 'IndentationError' in line:
                error_types['IndentationError'] += 1
            
            error_files[file_path].append(line.strip())
    
    return {
        'total_files': len(python_files),
        'files_with_errors': len(error_files),
        'total_errors': sum(len(errors) for errors in error_files.values()),
        'error_types': dict(error_types),
        'error_files': dict(error_files)
    }

def print_summary(results: Dict):
    """Print a summary of the scan results."""
    print("\n" + "=" * 60)
    print("QUICK ERROR SCAN SUMMARY")
    print("=" * 60)
    
    print(f"📊 Files processed: {results['total_files']}")
    print(f"📊 Files with errors: {results['files_with_errors']}")
    print(f"📊 Total errors: {results['total_errors']}")
    
    print(f"\n🔍 Error types:")
    for error_type, count in results['error_types'].most_common():
        percentage=(count / results['total_errors']) * 100 if results['total_errors'] > 0 else 0
        print(f"   {error_type}: {count} ({percentage:.1f}%)")

def print_files_by_error_count(results: Dict, top_n: int=20):
    """Print files sorted by error count."""
    print(f"\n📁 TOP {top_n} FILES WITH MOST ERRORS")
    print("-" * 50)
    
    # Sort files by error count
    sorted_files=sorted(
        results['error_files'].items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for i, (file_path, errors) in enumerate(sorted_files[:top_n], 1):
        absolute_path=os.path.abspath(file_path)
        print(f"{i:2d}. {file_path} ({len(errors)} errors)")
        print(f"     Location: {absolute_path}")
        
        # Show error types for this file
        file_error_types=Counter()
        for error in errors:
            if 'SyntaxError' in error:
                file_error_types['SyntaxError'] += 1
            elif 'IndentationError' in error:
                file_error_types['IndentationError'] += 1
        
        error_summary=', '.join([f"{t}: {c}" for t, c in file_error_types.most_common()])
        print(f"     Types: {error_summary}")

def print_files_by_directory(results: Dict):
    """Print files grouped by directory."""
    print(f"\n📁 FILES BY DIRECTORY")
    print("-" * 50)
    
    dir_files=defaultdict(list)
    for file_path, errors in results['error_files'].items():
        directory=os.path.dirname(file_path) or '.'
        dir_files[directory].append((file_path, len(errors)))
    
    # Sort directories by total errors
    sorted_dirs=sorted(
        dir_files.items(),
        key=lambda x: sum(count for _, count in x[1]),
        reverse=True
    )
    
    for directory, files in sorted_dirs:
        total_errors=sum(count for _, count in files)
        print(f"\n{directory}/ ({total_errors} total errors):")
        
        # Sort files in this directory by error count
        sorted_files=sorted(files, key=lambda x: x[1], reverse=True)
        for file_path, count in sorted_files[:5]:  # Show top 5
            print(f"  - {os.path.basename(file_path)} ({count} errors)")
        
        if len(sorted_files) > 5:
            print(f"  ... and {len(sorted_files) - 5} more files")

def print_priority_list(results: Dict):
    """Print a priority list for fixing errors."""
    print(f"\n🎯 PRIORITY FIXING LIST")
    print("-" * 50)
    print("Files to fix first (most errors or critical locations):")
    
    # Sort by error count and prioritize certain directories
    priority_files=[]
    for file_path, errors in results['error_files'].items():
        error_count=len(errors)
        priority_score=error_count
        
        # Boost priority for files in important directories
        if file_path.startswith('src/'):
            priority_score *= 2
        if 'training' in file_path or 'pipeline' in file_path:
            priority_score *= 1.5
        if file_path.startswith('scripts/'):
            priority_score *= 0.8  # Lower priority for scripts
        
        priority_files.append((file_path, error_count, priority_score))
    
    # Sort by priority score
    priority_files.sort(key=lambda x: x[2], reverse=True)
    
    for i, (file_path, error_count, priority_score) in enumerate(priority_files[:15], 1):
        print(f"{i:2d}. {file_path} ({error_count} errors, priority: {priority_score:.1f})")

def main():
    """Main function."""
    print("🚀 Quick Syntax Error Scanner")
    print("=" * 60)
    
    # Perform scan
    results=quick_scan()
    
    # Print results
    print_summary(results)
    print_files_by_error_count(results, top_n=15)
    print_files_by_directory(results)
    print_priority_list(results)
    
    print(f"\n✅ Scan completed!")
    print(f"💡 Tip: Focus on files in src/ directory first for maximum impact")

if __name__== "__main__":
    main()
