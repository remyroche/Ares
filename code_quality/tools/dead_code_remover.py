#!/usr/bin/env python3
"""
Dead Code Removal Tool
Identifies and removes unused functions, classes, and variables from Python files.
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Any
import argparse


class DeadCodeRemover:
    """Removes dead code from Python files."""
    
    def __init__(self):
        self.removals = 0
        self.files_processed = 0
        
    def remove_dead_code(self, filepath: str, dry_run: bool = True) -> bool:
        """Remove dead code from a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find all defined names
            defined_names = self._find_defined_names(tree)
            
            # Find all used names
            used_names = self._find_used_names(tree, content)
            
            # Find unused definitions
            unused_definitions = defined_names - used_names
            
            if not unused_definitions:
                return False
            
            # Remove unused definitions
            lines = content.split('\n')
            lines_to_remove = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name in unused_definitions:
                        # Don't remove if it's a main function or __init__
                        if node.name in ['main', '__init__', 'if __name__ == "__main__"']:
                            continue
                        
                        # Find the line range for this definition
                        start_line = node.lineno - 1
                        end_line = self._find_end_line(node, lines)
                        
                        for i in range(start_line, end_line + 1):
                            lines_to_remove.add(i)
            
            if lines_to_remove:
                if dry_run:
                    print(f"\n{filepath}:")
                    for line_idx in sorted(lines_to_remove):
                        if line_idx < len(lines):
                            print(f"  Would remove line {line_idx + 1}: {lines[line_idx].strip()}")
                else:
                    # Remove lines in reverse order to maintain line numbers
                    for line_idx in sorted(lines_to_remove, reverse=True):
                        if line_idx < len(lines):
                            print(f"Removing line {line_idx + 1}: {lines[line_idx].strip()}")
                            lines.pop(line_idx)
                    
                    # Write back the file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                
                self.removals += len(lines_to_remove)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return False
    
    def _find_defined_names(self, tree: ast.AST) -> Set[str]:
        """Find all defined names in the AST."""
        defined_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
        
        return defined_names
    
    def _find_used_names(self, tree: ast.AST, content: str) -> Set[str]:
        """Find all used names in the AST and content."""
        used_names = set()
        
        # Find names used in AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Find names used in strings (like decorators, etc.)
        for line in content.split('\n'):
            # Look for function calls in strings
            for name in re.findall(r'@(\w+)', line):
                used_names.add(name)
            for name in re.findall(r'(\w+)\(', line):
                used_names.add(name)
        
        return used_names
    
    def _find_end_line(self, node: ast.AST, lines: List[str]) -> int:
        """Find the end line of a node."""
        if hasattr(node, 'end_lineno'):
            return node.end_lineno - 1
        
        # Fallback: find the next non-indented line
        start_line = node.lineno - 1
        current_line = start_line + 1
        
        while current_line < len(lines):
            line = lines[current_line]
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break
            current_line += 1
        
        return current_line - 1
    
    def process_directory(self, directory: str, dry_run: bool = True) -> Dict[str, int]:
        """Process all Python files in a directory."""
        results = {'files_processed': 0, 'files_modified': 0, 'total_removals': 0}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    results['files_processed'] += 1
                    
                    if self.remove_dead_code(filepath, dry_run):
                        results['files_modified'] += 1
                        results['total_removals'] += self.removals
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Remove dead code from Python files')
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually remove code')
    parser.add_argument('--output', help='Output report to file')
    
    args = parser.parse_args()
    
    remover = DeadCodeRemover()
    results = remover.process_directory(args.directory, dry_run=not args.no_dry_run)
    
    report = f"""
Dead Code Removal Report
========================
Files processed: {results['files_processed']}
Files modified: {results['files_modified']}
Total lines removed: {results['total_removals']}
"""
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()