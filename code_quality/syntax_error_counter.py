#!/usr/bin/env python3
"""
Comprehensive Syntax Error Counter
Counts all syntax errors across the entire codebase.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

class SyntaxErrorCounter:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.syntax_errors = defaultdict(list)
        self.total_files = 0
        self.files_with_errors = 0
        self.total_errors = 0
        self.error_types = Counter()
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def count_syntax_errors(self, file_path):
        """Count syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse with AST
            try:
                ast.parse(content)
                return 0  # No syntax errors
                
            except SyntaxError as e:
                # Count this syntax error
                error_type = type(e).__name__
                self.error_types[error_type] += 1
                self.syntax_errors[file_path].append(str(e))
                return 1
                
            except Exception as e:
                # Other parsing errors
                error_type = type(e).__name__
                self.error_types[error_type] += 1
                self.syntax_errors[file_path].append(str(e))
                return 1
                
        except Exception as e:
            # File reading errors
            error_type = type(e).__name__
            self.error_types[error_type] += 1
            self.syntax_errors[file_path].append(f"File error: {e}")
            return 1
    
    def analyze_all_files(self):
        """Analyze all Python files for syntax errors."""
        python_files = self.find_python_files()
        self.total_files = len(python_files)
        
        print(f"🔍 Analyzing {self.total_files} Python files for syntax errors...")
        
        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                print(f"Processing file {i+1}/{self.total_files}...")
            
            error_count = self.count_syntax_errors(file_path)
            if error_count > 0:
                self.files_with_errors += 1
                self.total_errors += error_count
    
    def print_detailed_report(self):
        """Print a detailed syntax error report."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE SYNTAX ERROR REPORT")
        print(f"{'='*80}")
        print(f"📁 Total Python files: {self.total_files}")
        print(f"❌ Files with syntax errors: {self.files_with_errors}")
        print(f"🚨 Total syntax errors: {self.total_errors}")
        print(f"📊 Error rate: {(self.files_with_errors/self.total_files)*100:.1f}% of files have errors")
        
        print(f"\n🔍 ERROR TYPE BREAKDOWN:")
        for error_type, count in self.error_types.most_common():
            print(f"   • {error_type}: {count} occurrences")
        
        print(f"\n📋 FILES WITH MOST ERRORS:")
        # Sort files by error count
        sorted_files = sorted(self.syntax_errors.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        
        for i, (file_path, errors) in enumerate(sorted_files[:20], 1):
            print(f"   {i:2d}. {file_path}: {len(errors)} errors")
            if i <= 10:  # Show first 10 error details
                for j, error in enumerate(errors[:3], 1):  # Show first 3 errors per file
                    print(f"       {j}. {error}")
                if len(errors) > 3:
                    print(f"       ... and {len(errors) - 3} more errors")
        
        if len(sorted_files) > 20:
            print(f"   ... and {len(sorted_files) - 20} more files with errors")
        
        print(f"\n💡 SYNTAX ERROR SUMMARY:")
        print(f"   • Average errors per problematic file: {self.total_errors/self.files_with_errors:.1f}")
        print(f"   • Files without errors: {self.total_files - self.files_with_errors}")
        print(f"   • Percentage of clean files: {((self.total_files - self.files_with_errors)/self.total_files)*100:.1f}%")
    
    def save_detailed_report(self, output_path="syntax_error_report.json"):
        """Save detailed syntax error report as JSON."""
        report = {
            "summary": {
                "total_files": self.total_files,
                "files_with_errors": self.files_with_errors,
                "total_errors": self.total_errors,
                "error_rate_percentage": (self.files_with_errors/self.total_files)*100
            },
            "error_types": dict(self.error_types),
            "files_with_errors": {
                str(k): v for k, v in self.syntax_errors.items()
            },
            "files_with_most_errors": [
                {
                    "file": str(file_path),
                    "error_count": len(errors),
                    "errors": errors[:5]  # First 5 errors per file
                }
                for file_path, errors in sorted(
                    self.syntax_errors.items(), 
                    key=lambda x: len(x[1]), 
                    reverse=True
                )
            ]
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {output_path}")
    
    def get_error_statistics(self):
        """Get key error statistics."""
        return {
            "total_files": self.total_files,
            "files_with_errors": self.files_with_errors,
            "total_errors": self.total_errors,
            "error_rate": (self.files_with_errors/self.total_files)*100,
            "clean_files": self.total_files - self.files_with_errors
        }

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Comprehensive syntax error analysis for: {root_dir}")
    
    counter = SyntaxErrorCounter(root_dir)
    counter.analyze_all_files()
    
    # Print detailed report
    counter.print_detailed_report()
    
    # Save detailed report
    counter.save_detailed_report()
    
    # Get statistics
    stats = counter.get_error_statistics()
    
    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"📊 Total Python files: {stats['total_files']}")
    print(f"❌ Files with syntax errors: {stats['files_with_errors']}")
    print(f"✅ Clean files: {stats['clean_files']}")
    print(f"🚨 Total syntax errors: {stats['total_errors']}")
    print(f"📈 Error rate: {stats['error_rate']:.1f}%")

if __name__ == "__main__":
    main()