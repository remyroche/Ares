#!/usr/bin/env python3
"""
Working Import Cleaner
Removes unused imports from Python files.
"""

import ast
import os
import re
from typing import List, Dict, Set
import argparse


class WorkingImportCleaner:
    """Removes unused imports from Python files."""
    
    def __init__(self):
        self.files_processed = 0
        self.files_cleaned = 0
        self.total_imports_removed = 0
        
    def clean_file(self, filepath: str, dry_run: bool = False) -> bool:
        """Clean unused imports from a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip files with syntax errors
            try:
                ast.parse(content)
            except SyntaxError:
                print(f"⚠️  Skipping {filepath} due to syntax errors")
                return False
            
            original_content = content
            cleaned_content = self._remove_unused_imports(content)
            
            if cleaned_content != original_content:
                if not dry_run:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"✅ Cleaned: {filepath}")
                else:
                    print(f"🔧 Would clean: {filepath}")
                self.files_cleaned += 1
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return False
    
    def _remove_unused_imports(self, content: str) -> str:
        """Remove unused imports from content."""
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            imports_to_remove = set()
            
            # Find all import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname or alias.name.split('.')[0]
                        if not self._is_import_used(import_name, content, tree):
                            imports_to_remove.add(node.lineno - 1)
                            
                elif isinstance(node, ast.ImportFrom):
                    # For from imports, check if any of the imported names are used
                    unused_names = []
                    for alias in node.names:
                        import_name = alias.asname or alias.name
                        if import_name != '*' and not self._is_import_used(import_name, content, tree):
                            unused_names.append(alias.name)
                    
                    # If all names in the from import are unused, mark the whole line
                    if len(unused_names) == len(node.names) and node.names[0].name != '*':
                        imports_to_remove.add(node.lineno - 1)
            
            if not imports_to_remove:
                return content
            
            # Remove imports in reverse order to maintain line numbers
            for line_idx in sorted(imports_to_remove, reverse=True):
                if line_idx < len(lines):
                    lines.pop(line_idx)
                    self.total_imports_removed += 1
            
            return '\n'.join(lines)
            
        except Exception as e:
            print(f"Warning: Could not parse file for import cleaning: {e}")
            return content
    
    def _is_import_used(self, import_name: str, content: str, tree: ast.AST) -> bool:
        """Check if an import is actually used in the code."""
        # Check if used as a name
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == import_name:
                return True
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == import_name:
                return True
        
        # Check if used in strings (like type annotations, etc.)
        if f"'{import_name}'" in content or f'"{import_name}"' in content:
            return True
        
        # Check for indirect usage patterns
        if f"{import_name}." in content:
            return True
        
        return False
    
    def clean_directory(self, directory: str, dry_run: bool = False) -> Dict[str, int]:
        """Clean unused imports in all Python files in a directory."""
        stats = {'files_processed': 0, 'files_cleaned': 0, 'total_imports_removed': 0}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    stats['files_processed'] += 1
                    self.files_processed += 1
                    
                    if self.clean_file(filepath, dry_run):
                        stats['files_cleaned'] += 1
                        stats['total_imports_removed'] += self.total_imports_removed
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Remove unused imports from Python files')
    parser.add_argument('directory', help='Directory to clean')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without making changes')
    
    args = parser.parse_args()
    
    cleaner = WorkingImportCleaner()
    stats = cleaner.clean_directory(args.directory, args.dry_run)
    
    print(f"\n📊 Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files cleaned: {stats['files_cleaned']}")
    print(f"Total imports removed: {stats['total_imports_removed']}")


if __name__ == '__main__':
    main()