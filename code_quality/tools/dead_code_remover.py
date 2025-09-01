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

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="deadcoderemover initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DeadCodeRemover."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""Removes dead code from Python files."""
    
    def __init__(...):
    passself.removals = 0
        self.files_processed = 0
        
    def remove_dead_code(...) -> ...:
    """..."""
    passtry:
    passwith open(filepath, 'r', encoding='utf-8') as f:
    passcontent = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Find all defined names
            defined_names = self._find_defined_names(tree)
            
            # Find all used names
            used_names = self._find_used_names(tree, content)
            
            # Find unused definitions
            unused_definitions = defined_names - used_names
            
            if not unused_definitions:
    passreturn False
            
            # Remove unused definitions
            lines = content.split('\n')
            lines_to_remove = set()
            
            for node in ast.walk(tree):
    passif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    passif node.name in unused_definitions:
    pass# Don't remove if it's a main function or __init__
                        if node.name in ['main', '__init__', 'if __name__ == "__main__"']:
    passcontinue
                        
                        # Find the line range for this definition
                        start_line = node.lineno - 1
                        end_line = self._find_end_line(node, lines)
                        
                        for i in range(start_line, end_line + 1):
    passlines_to_remove.add(i)
            
            if lines_to_remove:
    passif dry_run:
    passprint(f"\n{filepath}:")
                    for line_idx in sorted(lines_to_remove):
    passif line_idx < len(lines):
    passprint(f"  Would remove line {line_idx + 1}: {lines[line_idx].strip()}")
                else:
    pass# Remove lines in reverse order to maintain line numbers
                    for line_idx in sorted(lines_to_remove, reverse=True):
    passif line_idx < len(lines):
    passprint(f"Removing line {line_idx + 1}: {lines[line_idx].strip()}")
                            lines.pop(line_idx)
                    
                    # Write back the file
                    with open(filepath, 'w', encoding='utf-8') as f:
    passf.write('\n'.join(lines))
                
                self.removals += len(lines_to_remove)
                return True
            
            return False
            
        except Exception as e:
    passpasspasspasspasspasspassprint(f"Error processing {filepath}: {e}")
            return False
    
    def _find_defined_names(...) -> ...:
    """..."""
    passdefined_names = set()
        
        for node in ast.walk(tree):
    passif isinstance(node, ast.FunctionDef):
    passdefined_names.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
    passpassdefined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
    passpassdefined_names.add(node.name)
            elif isinstance(node, ast.Assign):
    passpassfor target in node.targets:
    passif isinstance(target, ast.Name):
    passdefined_names.add(target.id)
        
        return defined_names
    
    def _find_used_names(...) -> ...:
    """..."""
    passused_names = set()
        
        # Find names used in AST
        for node in ast.walk(tree):
    passif isinstance(node, ast.Name):
    passused_names.add(node.id)
            elif isinstance(node, ast.Attribute):
    passpassif isinstance(node.value, ast.Name):
    passused_names.add(node.value.id)
        
        # Find names used in strings (like decorators, etc.)
        for line in content.split('\n'):
    pass# Look for function calls in strings
            for name in re.findall(r'@(\w+)', line):
    passused_names.add(name)
            for name in re.findall(r'(\w+)\(', line):
    passused_names.add(name)
        
        return used_names
    
    def _find_end_line(...) -> ...:
    """..."""
    passif hasattr(node, 'end_lineno'):
    passreturn node.end_lineno - 1
        
        # Fallback: find the next non-indented line
        start_line = node.lineno - 1
        current_line = start_line + 1
        
        while current_line < len(lines):
    passline = lines[current_line]
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
    passbreak
            current_line += 1
        
        return current_line - 1
    
    def process_directory(...) -> ...:
    """..."""
    passresults = {'files_processed': 0, 'files_modified': 0, 'total_removals': 0}
        
        for root, dirs, files in os.walk(directory):
    passfor file in files:
    passif file.endswith('.py'):
    passfilepath = os.path.join(root, file)
                    results['files_processed'] += 1
                    
                    if self.remove_dead_code(filepath, dry_run):
    passresults['files_modified'] += 1
                        results['total_removals'] += self.removals
        
        return results


def main(...):
    passparser = argparse.ArgumentParser(description='Remove dead code from Python files')
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
    passwith open(args.output, 'w') as f:
    passf.write(report)
        print(f"Report written to {args.output}")
    else:
    passprint(report)


if __name__ == '__main__':
    passmain()