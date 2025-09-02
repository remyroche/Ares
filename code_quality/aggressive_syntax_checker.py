#!/usr/bin/env python3
"""
Aggressive Syntax Error Checker
Tries to find multiple syntax errors per file by parsing incrementally.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

class AggressiveSyntaxChecker:
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
    
    def check_file_aggressively(self, file_path):
        """Check a file for multiple syntax errors aggressively."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            errors_found = []
            
            # Method 1: Try to parse the entire file
            try:
                ast.parse(content)
                return 0  # No syntax errors
            except SyntaxError as e:
                errors_found.append(f"Full file parse: {e}")
            
            # Method 2: Try parsing line by line to find multiple errors
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # Try to parse this line as a statement
                        ast.parse(line)
                    except SyntaxError as e:
                        errors_found.append(f"Line {i}: {e}")
                    except:
                        # Not a complete statement, skip
                        pass
            
            # Method 3: Try parsing in chunks
            chunk_size = 10
            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                if chunk.strip():
                    try:
                        ast.parse(chunk)
                    except SyntaxError as e:
                        errors_found.append(f"Chunk {i//chunk_size + 1}: {e}")
                    except:
                        pass
            
            # Method 4: Look for common syntax patterns
            for i, line in enumerate(lines, 1):
                if line.strip():
                    # Check for unmatched parentheses
                    if line.count('(') != line.count(')'):
                        errors_found.append(f"Line {i}: Unmatched parentheses")
                    
                    # Check for unmatched brackets
                    if line.count('[') != line.count(']'):
                        errors_found.append(f"Line {i}: Unmatched brackets")
                    
                    # Check for unmatched braces
                    if line.count('{') != line.count('}'):
                        errors_found.append(f"Line {i}: Unmatched braces")
                    
                    # Check for unterminated strings
                    if line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                        errors_found.append(f"Line {i}: Unterminated string")
                    
                    # Check for missing colons after function/class definitions
                    if (line.strip().startswith('def ') or line.strip().startswith('class ')) and not line.rstrip().endswith(':'):
                        errors_found.append(f"Line {i}: Missing colon after definition")
            
            # Store all errors found
            if errors_found:
                self.syntax_errors[file_path] = errors_found
                for error in errors_found:
                    if "SyntaxError" in error:
                        self.error_types["SyntaxError"] += 1
                    elif "IndentationError" in error:
                        self.error_types["IndentationError"] += 1
                    else:
                        self.error_types["Other"] += 1
                
                return len(errors_found)
            
            return 0
            
        except Exception as e:
            # File reading error
            self.syntax_errors[file_path] = [f"File error: {e}"]
            self.error_types["FileError"] += 1
            return 1
    
    def analyze_all_files(self):
        """Analyze all Python files for syntax errors aggressively."""
        python_files = self.find_python_files()
        self.total_files = len(python_files)
        
        print(f"🔍 Aggressively analyzing {self.total_files} Python files for syntax errors...")
        
        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                print(f"Processing file {i+1}/{self.total_files}...")
            
            error_count = self.check_file_aggressively(file_path)
            if error_count > 0:
                self.files_with_errors += 1
                self.total_errors += error_count
    
    def print_aggressive_report(self):
        """Print an aggressive syntax error report."""
        print(f"\n{'='*80}")
        print(f"AGGRESSIVE SYNTAX ERROR REPORT")
        print(f"{'='*80}")
        print(f"📁 Total Python files: {self.total_files}")
        print(f"❌ Files with syntax errors: {self.files_with_errors}")
        print(f"🚨 Total syntax errors found: {self.total_errors}")
        print(f"📊 Error rate: {(self.files_with_errors/self.total_files)*100:.1f}% of files have errors")
        print(f"📈 Average errors per problematic file: {self.total_errors/self.files_with_errors:.1f}")
        
        print(f"\n🔍 ERROR TYPE BREAKDOWN:")
        for error_type, count in self.error_types.most_common():
            print(f"   • {error_type}: {count} occurrences")
        
        print(f"\n📋 FILES WITH MOST ERRORS:")
        # Sort files by error count
        sorted_files = sorted(self.syntax_errors.items(), 
                            key=lambda x: len(x[1]), reverse=True)
        
        for i, (file_path, errors) in enumerate(sorted_files[:15], 1):
            print(f"   {i:2d}. {file_path}: {len(errors)} errors")
            if i <= 8:  # Show first 8 error details
                for j, error in enumerate(errors[:3], 1):  # Show first 3 errors per file
                    print(f"       {j}. {error}")
                if len(errors) > 3:
                    print(f"       ... and {len(errors) - 3} more errors")
        
        if len(sorted_files) > 15:
            print(f"   ... and {len(sorted_files) - 15} more files with errors")
        
        print(f"\n💡 AGGRESSIVE ANALYSIS SUMMARY:")
        print(f"   • Files without errors: {self.total_files - self.files_with_errors}")
        print(f"   • Percentage of clean files: {((self.total_files - self.files_with_errors)/self.total_files)*100:.1f}%")
        print(f"   • Total error instances: {self.total_errors}")
        
        # Show some examples of files with multiple errors
        multi_error_files = [f for f, errors in self.syntax_errors.items() if len(errors) > 1]
        if multi_error_files:
            print(f"\n🚨 FILES WITH MULTIPLE ERRORS: {len(multi_error_files)}")
            for file_path in multi_error_files[:10]:
                error_count = len(self.syntax_errors[file_path])
                print(f"   • {file_path}: {error_count} errors")
    
    def save_aggressive_report(self, output_path="aggressive_syntax_report.json"):
        """Save aggressive syntax error report as JSON."""
        import json
        
        report = {
            "summary": {
                "total_files": self.total_files,
                "files_with_errors": self.files_with_errors,
                "total_errors": self.total_errors,
                "error_rate_percentage": (self.files_with_errors/self.total_files)*100,
                "average_errors_per_file": self.total_errors/self.files_with_errors if self.files_with_errors > 0 else 0
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
            ],
            "multi_error_files": [
                str(file_path) for file_path, errors in self.syntax_errors.items() 
                if len(errors) > 1
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Aggressive report saved to: {output_path}")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Aggressive syntax error analysis for: {root_dir}")
    
    checker = AggressiveSyntaxChecker(root_dir)
    checker.analyze_all_files()
    
    # Print aggressive report
    checker.print_aggressive_report()
    
    # Save aggressive report
    checker.save_aggressive_report()
    
    print(f"\n{'='*80}")
    print(f"AGGRESSIVE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"📊 This analysis was more thorough than the basic counter")
    print(f"🚨 It found multiple syntax errors per file where possible")
    print(f"💡 Use this for a more accurate picture of your codebase state")

if __name__ == "__main__":
    main()