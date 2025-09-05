#!/usr/bin/env python3
"""
Automated Code Quality Fixes

This script implements automated fixes for the identified patterns.
"""

import ast
import re
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse

class CodeQualityAutoFixer:
    """Automated code quality fixer based on identified patterns."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_processed = 0
        self.errors = []
        
    def fix_unused_parameters(self, file_path: Path) -> int:
        """Fix unused parameters by adding underscore prefix."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            fixes = 0
            
            class UnusedParameterFixer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Check for unused parameters
                    for arg in node.args.args:
                        if not self._is_parameter_used(node, arg.arg):
                            # Add underscore prefix to unused parameters
                            if not arg.arg.startswith('_'):
                                arg.arg = f"_{arg.arg}"
                                fixes += 1
                    return node
                
                def _is_parameter_used(self, node, param_name):
                    """Check if parameter is used in function body."""
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and child.id == param_name:
                            return True
                    return False
            
            fixer = UnusedParameterFixer()
            new_tree = fixer.visit(tree)
            
            if fixes > 0:
                new_content = ast.unparse(new_tree)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixes_applied += fixes
                print(f"  Fixed {fixes} unused parameters in {file_path.name}")
            
            return fixes
            
        except Exception as e:
            self.errors.append(f"Error fixing unused parameters in {file_path}: {e}")
            return 0
    
    def fix_potential_none_access(self, file_path: Path) -> int:
        """Add None checks for potential None access."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixes = 0
            
            # Pattern: variable.method() or variable.attribute
            none_access_pattern = r'(\w+)\.(\w+)\('
            
            for i, line in enumerate(lines):
                # Skip comments and strings
                if line.strip().startswith('#') or '"""' in line or "'''" in line:
                    continue
                
                # Find potential None access patterns
                matches = re.finditer(none_access_pattern, line)
                for match in matches:
                    var_name = match.group(1)
                    method_name = match.group(2)
                    
                    # Check if this variable could be None
                    if self._variable_could_be_none(lines, i, var_name):
                        # Add None check
                        indent = len(line) - len(line.lstrip())
                        check_line = ' ' * indent + f"if {var_name} is not None:\n"
                        new_line = ' ' * (indent + 4) + line.strip()
                        
                        lines[i] = check_line + new_line
                        fixes += 1
                        break  # Only fix one per line to avoid conflicts
            
            if fixes > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                self.fixes_applied += fixes
                print(f"  Added {fixes} None checks in {file_path.name}")
            
            return fixes
            
        except Exception as e:
            self.errors.append(f"Error fixing None access in {file_path}: {e}")
            return 0
    
    def _variable_could_be_none(self, lines: List[str], line_index: int, var_name: str) -> bool:
        """Check if a variable could be None based on context."""
        # Look for assignments that could result in None
        for i in range(max(0, line_index - 10), line_index):
            line = lines[i]
            if f"{var_name} =" in line:
                # Check if assignment could be None
                if any(none_indicator in line for none_indicator in ['None', 'null', 'get(', 'find(']):
                    return True
        return False
    
    def fix_variable_shadowing(self, file_path: Path) -> int:
        """Fix variable shadowing by renaming variables."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            fixes = 0
            
            class VariableShadowingFixer(ast.NodeTransformer):
                def __init__(self):
                    self.scope_stack = []
                    self.variable_names = set()
                
                def visit_FunctionDef(self, node):
                    # Track function parameters
                    param_names = {arg.arg for arg in node.args.args}
                    self.scope_stack.append(param_names)
                    
                    # Check for shadowing in function body
                    for child in node.body:
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name):
                                    if target.id in self.variable_names:
                                        # Rename shadowed variable
                                        target.id = f"{target.id}_local"
                                        fixes += 1
                                    self.variable_names.add(target.id)
                    
                    self.scope_stack.pop()
                    return node
            
            fixer = VariableShadowingFixer()
            new_tree = fixer.visit(tree)
            
            if fixes > 0:
                new_content = ast.unparse(new_tree)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixes_applied += fixes
                print(f"  Fixed {fixes} variable shadowing issues in {file_path.name}")
            
            return fixes
            
        except Exception as e:
            self.errors.append(f"Error fixing variable shadowing in {file_path}: {e}")
            return 0
    
    def add_input_validation(self, file_path: Path) -> int:
        """Add input validation decorators."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixes = 0
            
            # Find functions that need validation
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and '(' in line:
                    # Check if function has parameters
                    if 'self' not in line or line.count(',') > 0:
                        # Add validation decorator
                        indent = len(line) - len(line.lstrip())
                        decorator = ' ' * indent + '@validate_inputs\n'
                        lines.insert(i, decorator)
                        fixes += 1
            
            if fixes > 0:
                # Add import for validation decorator
                if 'from validation import validate_inputs' not in content:
                    lines.insert(0, 'from validation import validate_inputs\n')
                    fixes += 1
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                self.fixes_applied += fixes
                print(f"  Added {fixes} input validation decorators in {file_path.name}")
            
            return fixes
            
        except Exception as e:
            self.errors.append(f"Error adding input validation in {file_path}: {e}")
            return 0
    
    def process_file(self, file_path: Path) -> Dict[str, int]:
        """Process a single Python file with all fixes."""
        if not file_path.suffix == '.py':
            return {}
        
        print(f"Processing {file_path.name}...")
        self.files_processed += 1
        
        fixes = {
            'unused_parameters': self.fix_unused_parameters(file_path),
            'none_access': self.fix_potential_none_access(file_path),
            'variable_shadowing': self.fix_variable_shadowing(file_path),
            'input_validation': self.add_input_validation(file_path)
        }
        
        return fixes
    
    def process_directory(self, directory: Path = None) -> Dict[str, int]:
        """Process all Python files in directory."""
        if directory is None:
            directory = self.project_root
        
        total_fixes = defaultdict(int)
        
        # Find all Python files
        python_files = list(directory.rglob('*.py'))
        
        # Exclude certain directories
        exclude_dirs = {'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'}
        python_files = [f for f in python_files if not any(excluded in f.parts for excluded in exclude_dirs)]
        
        print(f"Found {len(python_files)} Python files to process...")
        
        for file_path in python_files:
            try:
                fixes = self.process_file(file_path)
                for fix_type, count in fixes.items():
                    total_fixes[fix_type] += count
            except Exception as e:
                self.errors.append(f"Error processing {file_path}: {e}")
        
        return dict(total_fixes)
    
    def generate_report(self, total_fixes: Dict[str, int]):
        """Generate a summary report."""
        print("\n" + "="*60)
        print("AUTOMATED FIXES SUMMARY")
        print("="*60)
        print(f"Files processed: {self.files_processed}")
        print(f"Total fixes applied: {self.fixes_applied}")
        print(f"Errors encountered: {len(self.errors)}")
        
        print("\nFixes by type:")
        for fix_type, count in total_fixes.items():
            print(f"  {fix_type}: {count}")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Automated Code Quality Fixes')
    parser.add_argument('--project-root', default='/workspace', help='Project root directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without making changes')
    
    args = parser.parse_args()
    
    print("AUTOMATED CODE QUALITY FIXES")
    print("="*60)
    print(f"Project root: {args.project_root}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        return
    
    fixer = CodeQualityAutoFixer(args.project_root)
    total_fixes = fixer.process_directory()
    fixer.generate_report(total_fixes)

if __name__ == "__main__":
    main()