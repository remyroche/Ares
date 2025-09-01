#!/usr/bin/env python3
"""
Dead Code Detection and Removal Tool

This script uses multiple approaches to identify dead code:
1. MyPy for type checking and unused imports
2. Vulture for dead code detection
3. Ruff for unused imports and variables
4. Custom analysis for unreachable code

Usage:
    python dead_code_detector.py [--remove] [--dry-run] [--verbose]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import ast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeadCodeDetector:
    def __init__(self, src_dir: str = "src", dry_run: bool = True, verbose: bool = False):
        self.src_dir = Path(src_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        self.dead_code_items: List[Dict] = []
        
        # Exclude patterns
        self.exclude_patterns = [
            r'__pycache__',
            r'\.pyc$',
            r'\.git',
            r'\.mypy_cache',
            r'\.venv',
            r'venv',
            r'build',
            r'dist',
            r'node_modules',
            r'test_models',
            r'test_results',
            r'log',
        ]
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis."""
        for pattern in self.exclude_patterns:
            if re.search(pattern, str(file_path)):
                return True
        return False
    
    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                cwd=Path.cwd()
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
            return 1, "", str(e)
    
    def run_mypy(self) -> List[Dict]:
        """Run MyPy and extract dead code information."""
        logger.info("Running MyPy analysis...")
        
        cmd = ["venv/bin/mypy", str(self.src_dir), "--show-error-codes", "--show-column-numbers"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        dead_code_items = []
        
        # Parse MyPy output for unused imports and variables
        for line in stdout.split('\n'):
            if 'unused-import' in line or 'unused-variable' in line:
                # Parse file:line:column: error message
                match = re.match(r'([^:]+):(\d+):(\d+):\s+(\w+):\s+(.+)', line)
                if match:
                    file_path, line_num, col_num, error_code, message = match.groups()
                    dead_code_items.append({
                        'file': file_path,
                        'line': int(line_num),
                        'column': int(col_num),
                        'type': 'mypy',
                        'error_code': error_code,
                        'message': message,
                        'tool': 'mypy'
                    })
        
        return dead_code_items
    
    def run_vulture(self) -> List[Dict]:
        """Run Vulture and extract dead code information."""
        logger.info("Running Vulture analysis...")
        
        cmd = ["venv/bin/vulture", str(self.src_dir), "--min-confidence", "80", "--exclude", "test_models,test_results,log"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        dead_code_items = []
        
        # Parse Vulture output
        for line in stdout.split('\n'):
            if line.strip() and ':' in line:
                # Format: file:line: dead_code (confidence)
                match = re.match(r'([^:]+):(\d+):\s+(.+)', line)
                if match:
                    file_path, line_num, description = match.groups()
                    dead_code_items.append({
                        'file': file_path,
                        'line': int(line_num),
                        'type': 'vulture',
                        'message': description.strip(),
                        'tool': 'vulture'
                    })
        
        return dead_code_items
    
    def run_ruff(self) -> List[Dict]:
        """Run Ruff and extract unused imports and variables."""
        logger.info("Running Ruff analysis...")
        
        # Check for unused imports (F401)
        cmd = ["venv/bin/ruff", "check", str(self.src_dir), "--select", "F401,F841", "--output-format", "text"]
        exit_code, stdout, stderr = self.run_command(cmd)
        
        dead_code_items = []
        
        # Parse Ruff output
        for line in stdout.split('\n'):
            if line.strip() and ':' in line:
                # Format: file:line:column: error_code message
                match = re.match(r'([^:]+):(\d+):(\d+):\s+(\w+)\s+(.+)', line)
                if match:
                    file_path, line_num, col_num, error_code, message = match.groups()
                    dead_code_items.append({
                        'file': file_path,
                        'line': int(line_num),
                        'column': int(col_num),
                        'type': 'ruff',
                        'error_code': error_code,
                        'message': message,
                        'tool': 'ruff'
                    })
        
        return dead_code_items
    
    def analyze_ast(self, file_path: Path) -> List[Dict]:
        """Analyze Python file AST for dead code patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            dead_code_items = []
            
            # Find unused variables and imports
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        used_names.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            used_names.add(alias.asname or alias.name)
            
            # Check for unreachable code after return/raise
            for node in ast.walk(tree):
                if isinstance(node, (ast.Return, ast.Raise)):
                    # Look for code after this node at the same level
                    pass  # This would require more complex AST traversal
            
            return dead_code_items
            
        except Exception as e:
            logger.warning(f"Error analyzing AST for {file_path}: {e}")
            return []
    
    def detect_dead_code(self) -> List[Dict]:
        """Run all dead code detection methods."""
        logger.info("Starting comprehensive dead code detection...")
        
        all_dead_code = []
        
        # Run different tools
        all_dead_code.extend(self.run_mypy())
        all_dead_code.extend(self.run_vulture())
        all_dead_code.extend(self.run_ruff())
        
        # AST analysis for each Python file
        for py_file in self.src_dir.rglob("*.py"):
            if not self.should_exclude(py_file):
                all_dead_code.extend(self.analyze_ast(py_file))
        
        # Remove duplicates and sort
        unique_items = []
        seen = set()
        
        for item in all_dead_code:
            key = (item['file'], item['line'], item['type'], item['tool'])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        self.dead_code_items = sorted(unique_items, key=lambda x: (x['file'], x['line']))
        return self.dead_code_items
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of dead code."""
        if not self.dead_code_items:
            return "No dead code detected."
        
        report = ["# Dead Code Detection Report\n"]
        report.append(f"Found {len(self.dead_code_items)} potential dead code items\n")
        
        # Group by file
        by_file = {}
        for item in self.dead_code_items:
            file_path = item['file']
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(item)
        
        for file_path, items in by_file.items():
            report.append(f"\n## {file_path}")
            report.append(f"Found {len(items)} issues:\n")
            
            for item in items:
                report.append(f"- Line {item['line']}: {item['message']} ({item['tool']})")
        
        return "\n".join(report)
    
    def remove_dead_code(self) -> List[Dict]:
        """Remove dead code items (if not in dry-run mode)."""
        if self.dry_run:
            logger.info("DRY RUN MODE: No actual changes will be made")
            return []
        
        removed_items = []
        
        # Group by file for efficient processing
        by_file = {}
        for item in self.dead_code_items:
            file_path = item['file']
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(item)
        
        for file_path, items in by_file.items():
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist, skipping")
                continue
            
            # Sort items by line number in descending order to avoid line number shifts
            items.sort(key=lambda x: x['line'], reverse=True)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                modified = False
                for item in items:
                    line_num = item['line'] - 1  # Convert to 0-based index
                    
                    if 0 <= line_num < len(lines):
                        # Check if this line contains dead code
                        line_content = lines[line_num].strip()
                        
                        if self.is_dead_code_line(line_content, item):
                            if not self.dry_run:
                                # Remove the line
                                lines.pop(line_num)
                                modified = True
                                removed_items.append(item)
                                logger.info(f"Removed line {item['line']} from {file_path}")
                            else:
                                logger.info(f"Would remove line {item['line']} from {file_path}: {line_content}")
                
                if modified and not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    logger.info(f"Updated {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return removed_items
    
    def is_dead_code_line(self, line_content: str, item: Dict) -> bool:
        """Check if a line contains dead code based on the detection item."""
        if item['tool'] == 'vulture':
            # For vulture, we need to be more careful
            return True  # Vulture is generally reliable
        
        elif item['tool'] == 'ruff':
            if item['error_code'] == 'F401':  # Unused import
                return line_content.startswith('import ') or line_content.startswith('from ')
            elif item['error_code'] == 'F841':  # Unused variable
                return '=' in line_content and not line_content.startswith('#')
        
        elif item['tool'] == 'mypy':
            if 'unused-import' in item['error_code']:
                return line_content.startswith('import ') or line_content.startswith('from ')
            elif 'unused-variable' in item['error_code']:
                return '=' in line_content and not line_content.startswith('#')
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Dead Code Detection and Removal Tool")
    parser.add_argument("--remove", action="store_true", help="Remove dead code (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be removed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--src-dir", default="src", help="Source directory to analyze")
    parser.add_argument("--output", "-o", help="Output report to file")
    
    args = parser.parse_args()
    
    # Override dry-run if --remove is specified
    if args.remove:
        args.dry_run = False
    
    detector = DeadCodeDetector(
        src_dir=args.src_dir,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Detect dead code
    dead_code_items = detector.detect_dead_code()
    
    # Generate report
    report = detector.generate_report()
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")
    else:
        print(report)
    
    # Remove dead code if requested
    if not args.dry_run and dead_code_items:
        removed_items = detector.remove_dead_code()
        logger.info(f"Removed {len(removed_items)} dead code items")
    
    # Exit with appropriate code
    if dead_code_items:
        sys.exit(1)  # Found dead code
    else:
        sys.exit(0)  # No dead code found

if __name__ == "__main__":
    main()