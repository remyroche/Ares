#!/usr/bin/env python3
"""
Syntax Error Scanner for Ares Repository

This script scans the entire repository and provides a detailed report of:
1. Files with syntax errors
2. Number of errors per file
3. Types of errors found
4. Summary statistics
"""

import os
import re
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SyntaxErrorScanner:
    """Comprehensive syntax error scanner."""
    
    def __init__(self):
        self.error_files = defaultdict(list)
        self.error_types = Counter()
        self.total_errors = 0
        self.files_processed = 0
        
    def scan_file(self, file_path: str) -> List[str]:
        """Scan a single file for syntax errors."""
        try:
            result = subprocess.run(
                ['python', '-m', 'py_compile', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Parse error output
                errors = []
                for line in result.stderr.split('\n'):
                    if line.strip() and ('SyntaxError' in line or 'IndentationError' in line):
                        errors.append(line.strip())
                return errors
            return []
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout scanning {file_path}")
            return ["TimeoutError: File took too long to compile"]
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            return [f"ScanError: {e}"]
    
    def parse_error_line(self, error_line: str) -> Tuple[str, str, str]:
        """Parse an error line to extract error type, message, and file info."""
        # Extract error type
        if 'SyntaxError:' in error_line:
            error_type = 'SyntaxError'
        elif 'IndentationError:' in error_line:
            error_type = 'IndentationError'
        elif 'ImportError:' in error_line:
            error_type = 'ImportError'
        elif 'NameError:' in error_line:
            error_type = 'NameError'
        else:
            error_type = 'UnknownError'
        
        # Extract error message
        if ':' in error_line:
            parts = error_line.split(':', 2)
            if len(parts) >= 2:
                error_message = parts[1].strip()
            else:
                error_message = error_line
        else:
            error_message = error_line
        
        # Extract file info
        file_match = re.search(r'File "([^"]+)"', error_line)
        if file_match:
            file_info = file_match.group(1)
        else:
            file_info = "Unknown file"
        
        return error_type, error_message, file_info
    
    def scan_directory(self, directory: str) -> Dict:
        """Scan all Python files in a directory."""
        logger.info(f"🔍 Scanning directory: {directory}")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(python_files)} Python files")
        
        # Scan each file
        for file_path in python_files:
            self.files_processed += 1
            errors = self.scan_file(file_path)
            
            if errors:
                self.error_files[file_path] = errors
                self.total_errors += len(errors)
                
                # Count error types
                for error in errors:
                    error_type, _, _ = self.parse_error_line(error)
                    self.error_types[error_type] += 1
        
        return {
            'files_processed': self.files_processed,
            'files_with_errors': len(self.error_files),
            'total_errors': self.total_errors,
            'error_types': dict(self.error_types)
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive error report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("SYNTAX ERROR SCAN REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Files processed: {self.files_processed}")
        report_lines.append(f"Files with errors: {len(self.error_files)}")
        report_lines.append(f"Total errors: {self.total_errors}")
        report_lines.append("")
        
        # Error types breakdown
        report_lines.append("🔍 ERROR TYPES BREAKDOWN")
        report_lines.append("-" * 40)
        for error_type, count in self.error_types.most_common():
            percentage = (count / self.total_errors) * 100 if self.total_errors > 0 else 0
            report_lines.append(f"{error_type}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Files with errors (sorted by error count)
        report_lines.append("📁 FILES WITH ERRORS")
        report_lines.append("-" * 40)
        
        # Sort files by number of errors (descending)
        sorted_files = sorted(
            self.error_files.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for file_path, errors in sorted_files:
            # Use absolute path for better clarity
            absolute_path = os.path.abspath(file_path)
            relative_path = os.path.relpath(file_path, '.')
            report_lines.append(f"\n{relative_path} ({len(errors)} errors):")
            report_lines.append(f"   Location: {absolute_path}")
            
            # Group errors by type for this file
            file_error_types = Counter()
            for error in errors:
                error_type, _, _ = self.parse_error_line(error)
                file_error_types[error_type] += 1
            
            # Show error type breakdown for this file
            for error_type, count in file_error_types.most_common():
                report_lines.append(f"  - {error_type}: {count}")
            
            # Show first few actual error messages
            for i, error in enumerate(errors[:3]):  # Show first 3 errors
                error_type, message, _ = self.parse_error_line(error)
                report_lines.append(f"    {i+1}. {error_type}: {message[:100]}...")
            
            if len(errors) > 3:
                report_lines.append(f"    ... and {len(errors) - 3} more errors")
        
        # Detailed error breakdown
        report_lines.append("\n" + "=" * 80)
        report_lines.append("DETAILED ERROR BREAKDOWN")
        report_lines.append("=" * 80)
        
        for file_path, errors in sorted_files:
            relative_path = os.path.relpath(file_path, '.')
            absolute_path = os.path.abspath(file_path)
            report_lines.append(f"\n{relative_path}:")
            report_lines.append(f"Location: {absolute_path}")
            report_lines.append("-" * len(relative_path))
            
            for i, error in enumerate(errors, 1):
                report_lines.append(f"{i:3d}. {error}")
        
        report = "\n".join(report_lines)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 Report written to: {output_file}")
        
        return report
    
    def get_files_by_error_count(self, min_errors: int = 1) -> List[Tuple[str, str, int]]:
        """Get files with at least min_errors errors, sorted by error count."""
        files = []
        for file_path, errors in self.error_files.items():
            if len(errors) >= min_errors:
                relative_path = os.path.relpath(file_path, '.')
                absolute_path = os.path.abspath(file_path)
                files.append((relative_path, absolute_path, len(errors)))
        
        return sorted(files, key=lambda x: x[2], reverse=True)
    
    def get_files_by_error_type(self, error_type: str) -> List[Tuple[str, str, int]]:
        """Get files with specific error type, sorted by error count."""
        files = []
        for file_path, errors in self.error_files.items():
            type_count = sum(1 for error in errors 
                           if self.parse_error_line(error)[0] == error_type)
            if type_count > 0:
                relative_path = os.path.relpath(file_path, '.')
                absolute_path = os.path.abspath(file_path)
                files.append((relative_path, absolute_path, type_count))
        
        return sorted(files, key=lambda x: x[2], reverse=True)


def main():
    """Main function to run the syntax error scanner."""
    logger.info("🚀 Starting syntax error scanner")
    
    scanner = SyntaxErrorScanner()
    
    # Scan the current directory
    results = scanner.scan_directory('.')
    
    # Print summary
    logger.info("📊 Scan Summary:")
    logger.info(f"   Files processed: {results['files_processed']}")
    logger.info(f"   Files with errors: {results['files_with_errors']}")
    logger.info(f"   Total errors: {results['total_errors']}")
    
    # Generate and display report
    report = scanner.generate_report('syntax_error_report.txt')
    
    # Print top 10 files with most errors
    print("\n" + "=" * 60)
    print("TOP 10 FILES WITH MOST ERRORS")
    print("=" * 60)
    
    top_files = scanner.get_files_by_error_count(min_errors=1)[:10]
    for i, (relative_path, absolute_path, error_count) in enumerate(top_files, 1):
        print(f"{i:2d}. {relative_path} ({error_count} errors)")
        print(f"    Location: {absolute_path}")
    
    # Print files by error type
    print("\n" + "=" * 60)
    print("FILES BY ERROR TYPE")
    print("=" * 60)
    
    for error_type in ['SyntaxError', 'IndentationError']:
        files = scanner.get_files_by_error_type(error_type)
        if files:
            print(f"\n{error_type} files:")
            for relative_path, absolute_path, count in files[:5]:  # Show top 5
                print(f"  - {relative_path} ({count} errors)")
                print(f"    Location: {absolute_path}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
    
    logger.info("✅ Syntax error scanning completed!")
    logger.info("📄 Detailed report saved to: syntax_error_report.txt")


if __name__ == "__main__":
    main()
